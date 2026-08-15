# SPDX-License-Identifier: GPL-3.0-or-later
#
# Astro Color Mixer for Siril
# Yannick Dutertre (Cuiv), 2026
# cuivlazygeek@gmail.com
#
# Original PixInsight Astro Color Mixer by Patrick Cosgrove
# (Cosgrove's Cosmos, contact@cosgrovescosmos.com).
#
# This Siril/PyQt6 port is shared with Patrick Cosgrove's explicit permission,
# including permission to use any aspect of his original code and methods.
# Credit for the initial instantiation and workflow belongs to Patrick Cosgrove.
#
# For any questions about this specific Siril script, ask Yannick as above.
#
# Workflow: multi-pass, hue-targeted color and luminance refinement for
# nonlinear RGB data.
#
# Version 1.1.9
#
# 1.0.0 Initial Release
# 1.0.1 Improve math parity with Patrick Cosgrove's PixInsight implementation
# 1.0.2 Add built-in adjustment-set presets compatible with JSON recipes
# 1.0.3 Hide per-band Mask Soften controls outside Starless mode
# 1.0.4 Add mask overlay preview, opacity, and coverage statistics
# 1.0.5 Add full-resolution Apply stage timing diagnostics
# 1.0.6 Use sparse per-band application for narrower masks
# 1.0.7 Stream Apply timing progress while processing
# 1.0.8 Potential fix for label truncation
# 1.0.9 Fix preview too bright for some images
# 1.1.0 Add per-band L/R/G/B curves
# 1.1.1 Add a ColorMask-style custom color range band with hue wheel
# 1.1.2 Switch to Gaussian mask softening rather than simple box blur
# 1.1.3 Prevent the window from closing while Apply is processing
# 1.1.4 Add functional Relaxed, Strong, and Very Strong star protection
# 1.1.5 Optimize memory usage via tile processing
# 1.1.6 Make low-saturation protection continuously adjustable
# 1.1.7 Add protection-safe mask softening for images with stars
# 1.1.8 Make the curves editor modeless for unobstructed previewing on KDE Plasma
# 1.1.9 Fix numeric step-up buttons and add preview fit-to-window

"""
AstroColorMixer for Siril
==================================

A Siril python script that ports the PixInsight AstroColorMixer script
by Patrick Cosgrove (Cosgrove's Cosmos, see
https://cosgrovescosmos.com/astro-color-mixer-web)

This allows direct color and luminance adjustments to nonlinear RGB images.

This is a best effort port of the original PixInsight script, created
with permission from Patrick Cosgrove.
"""

import json
import sys
import time
from copy import deepcopy

import numpy as np

import sirilpy as s
from sirilpy import SirilConnectionError

s.ensure_installed("PyQt6")
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QStyle, QStyleOption, QStyleOptionSpinBox,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

VERSION = "1.1.9"

EPS = 1e-6
POSITIVE_LUMINANCE_GAIN = 0.55
CURVE_RESPONSE_GAIN = 0.25
MASK_SOFTEN_CONTROL_MAX = 200.0
MASK_SOFTEN_EFFECTIVE_MAX = 200.0
CUSTOM_MASK_TYPES = ("Chrominance", "Lightness", "Linear")
STAR_MASK_STRENGTHS = ("Relaxed", "Strong", "Very Strong")
AXIS = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)

BAND_DEFS = [
    {"id": "red", "center": 0.0, "label": "Red / H-alpha", "color": "#db534b"},
    {"id": "orange", "center": 30.0, "label": "Orange / Galaxy Cores", "color": "#d8872f"},
    {"id": "yellow", "center": 60.0, "label": "Yellow / Warm Stars", "color": "#d8c43f"},
    {"id": "green", "center": 120.0, "label": "Green / Cast Control", "color": "#3ba05a"},
    {"id": "cyan", "center": 180.0, "label": "Cyan / OIII", "color": "#39b7b5"},
    {"id": "blue", "center": 240.0, "label": "Blue / Reflection Nebula", "color": "#4a76d4"},
    {"id": "purple", "center": 275.0, "label": "Purple / Violet Cleanup", "color": "#7a61d7"},
    {"id": "magenta", "center": 315.0, "label": "Magenta / Halo Cleanup", "color": "#cb4ca8"},
    {"id": "custom", "center": 0.0, "label": "Custom Color Range", "color": "#f0f0f0", "custom": True},
]

SENSITIVITY_RANGES = {
    "Fine": {"hueShift": 5.0, "saturation": 15.0, "luminance": 10.0, "neutral": 5.0},
    "Normal": {"hueShift": 20.0, "saturation": 60.0, "luminance": 30.0, "neutral": 20.0},
    "Advanced": {"hueShift": 45.0, "saturation": 100.0, "luminance": 60.0, "neutral": 50.0},
    "Strong": {"hueShift": 45.0, "saturation": 100.0, "luminance": 60.0, "neutral": 50.0},
}

PROTECTION_PRESETS = {
    "stars": {"satFloor": 0.05, "satFull": 0.25, "darkFloor": 0.04, "darkFull": 0.18, "highlightStart": 0.70, "highlightFull": 0.95},
    "starless": {"satFloor": 0.03, "satFull": 0.18, "darkFloor": 0.02, "darkFull": 0.12, "highlightStart": 0.85, "highlightFull": 0.98},
}

RANGE_PRESETS = {
    "All": (0.0, 1.0),
    "Shadows": (0.0, 0.35),
    "Midtones": (0.20, 0.78),
    "Highlights": (0.55, 1.0),
    "Bright Stars": (0.78, 1.0),
}


# ======================================================================
# Siril / Preview Helpers
# ======================================================================

def _to_hwc(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return data[:, :, None], "hw"
    if data.ndim != 3:
        raise ValueError(f"Unexpected pixel data with {data.ndim} dimensions.")
    d0, _d1, d2 = data.shape
    if d2 in (1, 3) and d2 != d0:
        return data, "hwc"
    if d0 in (1, 3) and d0 != d2:
        return np.moveaxis(data, 0, -1), "chw"
    return data, "hwc"


def _from_hwc(data, layout):
    if layout == "chw":
        return np.moveaxis(data, -1, 0)
    if layout == "hw":
        return data[:, :, 0]
    return data


def _normalize_range(data):
    data = np.asarray(data, dtype=np.float32)
    if data.size and float(np.nanmax(data)) > 1.5:
        return data / 65535.0
    return data


def _ensure_rgb(data):
    if data.ndim != 3:
        raise ValueError("Astro Color Mixer requires an image array.")
    if data.shape[2] == 3:
        return data.astype(np.float32, copy=False)
    if data.shape[2] == 1:
        return np.repeat(data, 3, axis=2).astype(np.float32, copy=False)
    raise ValueError(f"Astro Color Mixer requires a 1- or 3-channel image, got shape {data.shape}.")


def _downsample_area(data, max_dim):
    h, w = data.shape[:2]
    factor = max(1, int(np.ceil(max(h, w) / float(max_dim))))
    if factor == 1:
        return data, 1.0
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    pad_spec = ((0, pad_h), (0, pad_w)) + (((0, 0),) if data.ndim == 3 else ())
    padded = np.pad(np.asarray(data, dtype=np.float32), pad_spec, mode="edge")
    new_h = padded.shape[0] // factor
    new_w = padded.shape[1] // factor
    if data.ndim == 3:
        reduced = padded.reshape(new_h, factor, new_w, factor, data.shape[2]).mean(axis=(1, 3), dtype=np.float32)
    else:
        reduced = padded.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3), dtype=np.float32)
    return reduced.astype(np.float32, copy=False), 1.0 / factor


def _array_to_qpixmap(data, display_range=None):
    arr = np.asarray(data, dtype=np.float32)
    if display_range is not None:
        display_lo, display_hi = display_range
        arr = np.clip((arr - display_lo) / (display_hi - display_lo), 0.0, 1.0)
    if arr.ndim == 2 or arr.shape[2] == 1:
        gray = np.clip(arr[:, :, 0] if arr.ndim == 3 else arr, 0.0, 1.0)
        arr8 = (gray * 255.0 + 0.5).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        arr8 = np.clip(arr[:, :, :3], 0.0, 1.0)
        arr8 = (arr8 * 255.0 + 0.5).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w, _ = arr8.shape
        qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _mask_values_from_rgb(mask_rgb):
    arr = np.asarray(mask_rgb, dtype=np.float32)
    if arr.ndim == 2:
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    return np.clip(arr[:, :, 0], 0.0, 1.0).astype(np.float32)


def _mask_overlay_image(base_rgb, mask_values, color, opacity):
    base = np.clip(np.asarray(base_rgb, dtype=np.float32)[:, :, :3], 0.0, 1.0)
    mask = np.clip(np.asarray(mask_values, dtype=np.float32), 0.0, 1.0)
    tint = np.array(color, dtype=np.float32)[None, None, :]
    alpha = np.clip(float(opacity), 0.0, 1.0) * mask[:, :, None]
    return np.clip(base * (1.0 - alpha) + tint * alpha, 0.0, 1.0).astype(np.float32)


def _mask_stats_text(mask_values):
    mask = np.clip(np.asarray(mask_values, dtype=np.float32), 0.0, 1.0)
    if mask.size == 0:
        return "Mask: empty"
    mean = float(np.mean(mask))
    maxv = float(np.max(mask))
    p10 = float(np.mean(mask > 0.10) * 100.0)
    p25 = float(np.mean(mask > 0.25) * 100.0)
    p50 = float(np.mean(mask > 0.50) * 100.0)
    return f"Mask mean {mean:.3f} | max {maxv:.3f} | >10% {p10:.1f}% | >25% {p25:.1f}% | >50% {p50:.1f}%"


def _clean_curve_points(points):
    cleaned = []
    for p in points or []:
        try:
            x = float(p[0])
            y = float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            cleaned.append([float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))])
    if not any(abs(x) < 0.0005 for x, _y in cleaned):
        cleaned.append([0.0, 0.0])
    if not any(abs(x - 1.0) < 0.0005 for x, _y in cleaned):
        cleaned.append([1.0, 1.0])
    cleaned.sort(key=lambda pt: pt[0])

    deduped = []
    for x, y in cleaned:
        if deduped and abs(x - deduped[-1][0]) < 0.0005:
            deduped[-1] = [x, y]
            continue
        deduped.append([x, y])

    deduped[0] = [0.0, deduped[0][1]]
    deduped[-1] = [1.0, deduped[-1][1]]
    deduped.sort(key=lambda pt: pt[0])
    return deduped


def _identity_curve_points():
    return [[0.0, 0.0], [1.0, 1.0]]


CURVE_CHANNELS = ("L", "R", "G", "B")
CURVE_CHANNEL_LABELS = {"L": "Luminance", "R": "Red", "G": "Green", "B": "Blue"}
CURVE_CHANNEL_COLORS = {"L": "#ffc43a", "R": "#ff6b5f", "G": "#66d36e", "B": "#66a6ff"}


def _identity_curve_set():
    return {channel: {"points": _identity_curve_points()} for channel in CURVE_CHANNELS}


def _normalize_curve_set(curve_set=None, legacy_luminance_curve=None):
    out = _identity_curve_set()
    has_curve_set = isinstance(curve_set, dict)
    if has_curve_set:
        for channel in CURVE_CHANNELS:
            curve = curve_set.get(channel, {})
            if isinstance(curve, dict):
                out[channel] = {"points": _clean_curve_points(curve.get("points", _identity_curve_points()))}
            else:
                out[channel] = {"points": _clean_curve_points(curve)}
    if not has_curve_set and legacy_luminance_curve is not None:
        legacy = legacy_luminance_curve if isinstance(legacy_luminance_curve, dict) else {}
        out["L"] = {"points": _clean_curve_points(legacy.get("points", _identity_curve_points()))}
    return out


def _curve_is_identity(points):
    pts = _clean_curve_points(points)
    if len(pts) != 2:
        return False
    return abs(pts[0][0]) <= EPS and abs(pts[0][1]) <= EPS and abs(pts[1][0] - 1.0) <= EPS and abs(pts[1][1] - 1.0) <= EPS


def _curve_set_active(curve_set):
    curves = _normalize_curve_set(curve_set)
    return any(not _curve_is_identity(curves[channel]["points"]) for channel in CURVE_CHANNELS)


def _curve_lut(points, size=4096):
    pts = _clean_curve_points(points)
    x = np.array([p[0] for p in pts], dtype=np.float32)
    y = np.array([p[1] for p in pts], dtype=np.float32)
    dom = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
    if len(pts) <= 2:
        return np.interp(dom, x, y).astype(np.float32)

    h = np.diff(x)
    delta = np.diff(y) / np.maximum(h, EPS)
    slopes = np.zeros_like(y)
    slopes[0] = delta[0]
    slopes[-1] = delta[-1]
    for i in range(1, len(y) - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            slopes[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            slopes[i] = (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i]))

    seg = np.searchsorted(x, dom, side="right") - 1
    seg = np.clip(seg, 0, len(x) - 2)
    x0 = x[seg]
    x1 = x[seg + 1]
    y0 = y[seg]
    y1 = y[seg + 1]
    m0 = slopes[seg]
    m1 = slopes[seg + 1]
    dx = np.maximum(x1 - x0, EPS)
    t = (dom - x0) / dx
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    lut = h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1
    return np.clip(lut, 0.0, 1.0).astype(np.float32)


def _apply_curve_lut(values, lut):
    vals = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    idx = vals * (len(lut) - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, len(lut) - 1)
    frac = idx - lo
    return (lut[lo] * (1.0 - frac) + lut[hi] * frac).astype(np.float32)

def _box_blur(data, radius):
    radius = int(max(0, round(radius)))
    if radius <= 0:
        return data.copy()
    work = data[:, :, None] if data.ndim == 2 else data
    squeeze = data.ndim == 2
    work = np.asarray(work, dtype=np.float32)
    h, w = work.shape[:2]
    if h <= 1 and w <= 1:
        return data.copy()

    xs = np.arange(w)
    x0 = np.maximum(0, xs - radius)
    x1 = np.minimum(w - 1, xs + radius)
    csum = np.concatenate(
        [np.zeros((h, 1, work.shape[2]), dtype=np.float32), np.cumsum(work, axis=1, dtype=np.float32)],
        axis=1,
    )
    horiz = (csum[:, x1 + 1, :] - csum[:, x0, :]) / (x1 - x0 + 1)[None, :, None]

    ys = np.arange(h)
    y0 = np.maximum(0, ys - radius)
    y1 = np.minimum(h - 1, ys + radius)
    csum = np.concatenate(
        [np.zeros((1, w, work.shape[2]), dtype=np.float32), np.cumsum(horiz, axis=0, dtype=np.float32)],
        axis=0,
    )
    out = (csum[y1 + 1, :, :] - csum[y0, :, :]) / (y1 - y0 + 1)[:, None, None]
    out = clamp01(out).astype(np.float32)
    return out[:, :, 0] if squeeze else out


def _gaussian_box_radii(radius):
    """Return three box radii whose combined variance approximates a Gaussian."""
    radius = int(max(0, radius))
    if radius == 0:
        return ()
    sigma_sq = radius * (radius + 1.0) / 3.0
    count = 3
    ideal_width = np.sqrt(12.0 * sigma_sq / count + 1.0)
    lower_width = int(np.floor(ideal_width))
    if lower_width % 2 == 0:
        lower_width -= 1
    lower_width = max(1, lower_width)
    upper_width = lower_width + 2
    numerator = 12.0 * sigma_sq - count * lower_width * lower_width - 4.0 * count * lower_width - 3.0 * count
    lower_count = int(np.clip(round(numerator / (-4.0 * lower_width - 4.0)), 0, count))
    widths = [lower_width] * lower_count + [upper_width] * (count - lower_count)
    return tuple((width - 1) // 2 for width in widths if width > 1)


def _gaussian_blur_approx(data, radius):
    result = np.asarray(data, dtype=np.float32)
    for box_radius in _gaussian_box_radii(radius):
        result = _box_blur(result, box_radius)
    return result.astype(np.float32, copy=False)


def _effective_mask_soften_radius(value):
    """Map the UI value directly to the convolution radius in pixels."""
    return float(np.clip(value, 0.0, MASK_SOFTEN_CONTROL_MAX))


def _soften_mask(data, amount):
    """Apply continuous, rotationally smooth Gaussian-style convolution."""
    amount = float(np.clip(amount, 0.0, MASK_SOFTEN_EFFECTIVE_MAX))
    if amount <= EPS:
        return np.asarray(data, dtype=np.float32).copy()
    lower_radius = int(np.floor(amount))
    blend = amount - lower_radius
    lower = _gaussian_blur_approx(data, lower_radius)
    if blend <= EPS:
        return lower
    upper = _gaussian_blur_approx(data, lower_radius + 1)
    return (lower * (1.0 - blend) + upper * blend).astype(np.float32)

# ======================================================================
# Core ACM Math
# ======================================================================

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(edge0, edge1, x):
    if abs(edge1 - edge0) < EPS:
        return np.where(np.asarray(x) >= edge1, 1.0, 0.0).astype(np.float32)
    t = np.clip((np.asarray(x, dtype=np.float32) - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luma709(rgb):
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def rgb_to_hsl(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    d = mx - mn
    light = (mx + mn) * 0.5
    sat = np.zeros_like(light, dtype=np.float32)
    nz = d > EPS
    sat[nz] = np.where(light[nz] > 0.5, d[nz] / np.maximum(2.0 - mx[nz] - mn[nz], EPS), d[nz] / np.maximum(mx[nz] + mn[nz], EPS))
    hue = np.zeros_like(light, dtype=np.float32)
    rmax = nz & (mx == r)
    gmax = nz & (mx == g)
    bmax = nz & (mx == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / np.maximum(d[rmax], EPS)) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / np.maximum(d[gmax], EPS)) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / np.maximum(d[bmax], EPS)) + 4.0
    hue = (hue * 60.0) % 360.0
    return {"h": hue.astype(np.float32), "s": sat.astype(np.float32), "l": light.astype(np.float32), "y": luma709(rgb).astype(np.float32)}


def circular_hue_distance(hue, center):
    delta = np.abs((hue % 360.0) - (center % 360.0))
    return np.minimum(delta, 360.0 - delta)


def hue_mask(hue, center, width, feather):
    distance = circular_hue_distance(hue, center)
    outer = max(float(width), EPS)
    inner = outer * (1.0 - np.clip(float(feather), 0.0, 1.0))
    if outer - inner <= EPS:
        return (distance <= outer).astype(np.float32)
    t = np.clip((distance - inner) / (outer - inner), 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _hue_arc_position(hue, start, end):
    start = float(start) % 360.0
    end = float(end) % 360.0
    span = (end - start) % 360.0
    if span <= EPS:
        span = 360.0
    pos = ((np.asarray(hue, dtype=np.float32) - start) % 360.0) / span
    inside = pos <= 1.0
    return pos.astype(np.float32), inside


def custom_color_mask(source_hsl, band):
    params = band.get("customMask", {}) if isinstance(band.get("customMask", {}), dict) else {}
    start = float(np.clip(params.get("startHue", 300.0), 0.0, 360.0))
    end = float(np.clip(params.get("endHue", 60.0), 0.0, 360.0))
    strength = float(np.clip(params.get("strength", 0.75), 0.0, 1.0))
    feather = float(np.clip(params.get("feather", 0.0), 0.0, 1.0))
    mask_type = params.get("type", "Chrominance")

    pos, inside = _hue_arc_position(source_hsl["h"], start, end)
    full_circle = ((end - start) % 360.0) <= EPS
    if full_circle:
        profile = np.ones_like(pos, dtype=np.float32)
    else:
        triangle = np.clip(1.0 - np.abs(pos - 0.5) * 2.0, 0.0, 1.0)
        # PixInsight ColorMask shapes the hue profile with ~mtf(hue_distance,
        # 1-strength). Preserve the exact endpoints to avoid the 0/0 limits of
        # the midtones transfer function at minimum and maximum strength.
        balance = 1.0 - strength
        denominator = (2.0 * triangle - 1.0) * balance - triangle
        with np.errstate(divide="ignore", invalid="ignore"):
            profile = 1.0 - ((triangle - 1.0) * balance / denominator)
        profile = np.where(triangle <= EPS, 0.0, profile)
        profile = np.where(triangle >= 1.0 - EPS, 1.0, profile)
        if feather > EPS:
            edge_width = 0.5 * feather
            edge_taper = smoothstep(0.0, edge_width, pos) * smoothstep(0.0, edge_width, 1.0 - pos)
            profile *= edge_taper
    mask = np.where(inside, profile, 0.0).astype(np.float32)

    if mask_type == "Chrominance":
        mask *= np.power(np.clip(source_hsl["s"], 0.0, 1.0), 0.35)
    elif mask_type == "Lightness":
        mask *= np.power(np.clip(source_hsl["l"], 0.0, 1.0), 0.45)
    return clamp01(mask).astype(np.float32)


def compute_range_mask(luma, range_mask):
    if not range_mask.get("enabled", False):
        return np.ones_like(luma, dtype=np.float32)
    low = float(range_mask.get("low", 0.0))
    high = float(range_mask.get("high", 1.0))
    feather = float(range_mask.get("feather", 0.10))
    m = smoothstep(low - feather, low, luma) * (1.0 - smoothstep(high, high + feather, luma))
    if range_mask.get("boostEnabled", False):
        m = np.power(np.clip(m, 0.0, 1.0), 0.55).astype(np.float32)
    soften = _effective_mask_soften_radius(range_mask.get("maskSoftenRadius", 0) or 0)
    soften *= float(range_mask.get("_previewSpatialScale", 1.0))
    return clamp01(_soften_mask(m.astype(np.float32), soften)) if soften > EPS else clamp01(m).astype(np.float32)


def normalize_star_mask_strength(value):
    # Legacy Siril recipes stored a numeric slider value that always processed
    # as Strong. Keep those recipes visually compatible.
    if isinstance(value, str) and value in STAR_MASK_STRENGTHS:
        return value
    return "Strong"


def star_mask_strength_settings(value):
    strength = normalize_star_mask_strength(value)
    if strength == "Very Strong":
        return {
            "seedFloor": 0.13,
            "contrastLow": 0.018,
            "contrastHigh": 0.090,
            "luminanceLow": 0.13,
            "luminanceHigh": 0.46,
            "radius": 7,
            "suppress": 1.0,
            "blurRadius": 1,
        }
    if strength == "Relaxed":
        return {
            "seedFloor": 0.22,
            "contrastLow": 0.040,
            "contrastHigh": 0.160,
            "luminanceLow": 0.22,
            "luminanceHigh": 0.62,
            "radius": 3,
            "suppress": 0.92,
            "blurRadius": 0,
        }
    return {
        "seedFloor": 0.17,
        "contrastLow": 0.026,
        "contrastHigh": 0.115,
        "luminanceLow": 0.17,
        "luminanceHigh": 0.52,
        "radius": 5,
        "suppress": 0.97,
        "blurRadius": 1,
    }

def star_protection_mask(luma, strength):
    settings = star_mask_strength_settings(strength)
    h, w = luma.shape
    seeds = np.zeros_like(luma, dtype=np.float32)
    if h < 5 or w < 5:
        return seeds

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            center = float(luma[y, x])
            if center < settings["seedFloor"]:
                continue
            patch = luma[y - 2:y + 3, x - 2:x + 3]
            if center + 0.0001 < float(np.max(patch)):
                continue
            ring = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            contrast = center - float(np.mean(ring, dtype=np.float64))
            compact = smoothstep(settings["contrastLow"], settings["contrastHigh"], contrast) * smoothstep(settings["luminanceLow"], settings["luminanceHigh"], center)
            if compact > 0:
                seeds[y, x] = compact

    radius = int(settings["radius"])
    expanded = np.zeros_like(seeds, dtype=np.float32)
    offsets = []
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            dist = float(np.hypot(ox, oy))
            if dist <= radius:
                offsets.append((ox, oy, 1.0 - float(smoothstep(0.0, radius, dist))))

    ys, xs = np.nonzero(seeds > 0)
    for y, x in zip(ys, xs):
        seed = float(seeds[y, x])
        for ox, oy, falloff in offsets:
            ty = y + oy
            tx = x + ox
            if 0 <= ty < h and 0 <= tx < w:
                value = seed * falloff
                if value > expanded[ty, tx]:
                    expanded[ty, tx] = value

    blur_radius = int(settings["blurRadius"])
    return clamp01(_box_blur(expanded, blur_radius) if blur_radius > 0 else expanded).astype(np.float32)


def make_default_band(defn):
    return {
        "id": defn["id"], "center": defn["center"], "label": defn["label"], "color": defn["color"],
        "custom": bool(defn.get("custom", False)),
        "hueShift": 0.0, "saturation": 0.0, "luminance": 0.0, "luminanceMode": "slider",
        "luminanceCurve": {"points": _identity_curve_points()}, "curveSet": _identity_curve_set(),
        "width": 45.0, "feather": 0.75, "maskSoftenRadius": 0,
        "customMask": {"startHue": 300.0, "endHue": 60.0, "strength": 0.75, "feather": 0.0, "type": "Chrominance"},
    }


def default_pass(pass_id="pass-1", name="Base Pass"):
    return {
        "id": pass_id, "name": name, "label": name, "enabled": True, "selectedBandId": "red",
        "rangeMask": {"enabled": False, "low": 0.0, "high": 1.0, "feather": 0.10, "preset": "All", "maskSoftenRadius": 0, "boostEnabled": False},
        "neutralLuminance": {"luminance": 0.0, "satStart": 0.04, "satFull": 0.16},
        "bands": [make_default_band(d) for d in BAND_DEFS],
    }


def default_recipe():
    return {
        "version": "siril-acm-1.0",
        "imageType": "stars",
        "sensitivity": "Normal",
        "globalStrength": 1.0,
        "protectionControls": {"protectStars": True, "protectLowSaturation": 1.0, "starMaskStrength": "Strong"},
        "passes": [default_pass()],
    }

def _band(pass_state, band_id):
    return next(b for b in pass_state["bands"] if b["id"] == band_id)


def _set_band(pass_state, band_id, hue=None, saturation=None, luminance=None, width=None, feather=None, soften=None):
    band = _band(pass_state, band_id)
    if hue is not None:
        band["hueShift"] = float(hue)
    if saturation is not None:
        band["saturation"] = float(saturation)
    if luminance is not None:
        band["luminance"] = float(luminance)
    if width is not None:
        band["width"] = float(width)
    if feather is not None:
        band["feather"] = float(feather)
    if soften is not None:
        band["maskSoftenRadius"] = float(soften)


def _set_range(pass_state, enabled, low=0.0, high=1.0, feather=0.10, preset="Custom", boost=False, soften=0):
    pass_state["rangeMask"] = {
        "enabled": bool(enabled),
        "low": float(low),
        "high": float(high),
        "feather": float(feather),
        "preset": preset,
        "maskSoftenRadius": float(soften),
        "boostEnabled": bool(boost),
    }


def _preset_base(image_type="stars", sensitivity="Normal", protect_low_sat=True, global_strength=1.0):
    recipe = default_recipe()
    recipe["version"] = "siril-acm-preset-1.0"
    recipe["imageType"] = image_type
    recipe["sensitivity"] = sensitivity
    recipe["globalStrength"] = global_strength
    recipe["protectionControls"] = {
        "protectStars": image_type != "starless",
        "protectLowSaturation": 1.0 if protect_low_sat else 0.0,
        "starMaskStrength": "Strong",
    }
    return recipe


def _preset_galaxy_broadband():
    recipe = _preset_base("stars", "Normal", protect_low_sat=False, global_strength=0.85)
    p = default_pass("pass-1", "Galaxy Color Balance")
    p["selectedBandId"] = "orange"
    _set_band(p, "orange", saturation=14, luminance=4, width=50, feather=0.75)
    _set_band(p, "yellow", saturation=8, luminance=2, width=42, feather=0.80)
    _set_band(p, "blue", saturation=10, luminance=1, width=46, feather=0.80)
    _set_band(p, "green", saturation=-10, luminance=-1, width=55, feather=0.85)
    p["neutralLuminance"]["luminance"] = -2

    p2 = default_pass("pass-2", "Faint Arms and Dust")
    p2["selectedBandId"] = "blue"
    _set_range(p2, True, 0.18, 0.72, 0.12, "Midtones", boost=True)
    _set_band(p2, "orange", saturation=8, width=52, feather=0.85)
    _set_band(p2, "blue", saturation=8, width=50, feather=0.85)
    recipe["passes"] = [p, p2]
    return recipe


def _preset_hoo_nebula_separation():
    recipe = _preset_base("starless", "Normal", protect_low_sat=True, global_strength=0.9)
    p = default_pass("pass-1", "HOO Separation")
    p["selectedBandId"] = "cyan"
    _set_band(p, "red", hue=-3, saturation=16, luminance=4, width=48, feather=0.78, soften=1)
    _set_band(p, "cyan", hue=4, saturation=18, luminance=5, width=50, feather=0.78, soften=1)
    _set_band(p, "green", saturation=-12, luminance=-2, width=55, feather=0.85, soften=1)

    p2 = default_pass("pass-2", "Faint Nebula Luminance")
    p2["selectedBandId"] = "red"
    _set_range(p2, True, 0.10, 0.70, 0.16, "Midtones", boost=True, soften=1)
    _set_band(p2, "red", luminance=6, width=55, feather=0.85, soften=1)
    _set_band(p2, "cyan", luminance=5, width=55, feather=0.85, soften=1)
    recipe["passes"] = [p, p2]
    return recipe


def _preset_sho_palette_refinement():
    recipe = _preset_base("starless", "Advanced", protect_low_sat=True, global_strength=0.85)
    p = default_pass("pass-1", "SHO Palette Refinement")
    p["selectedBandId"] = "yellow"
    _set_band(p, "yellow", hue=-5, saturation=18, luminance=4, width=55, feather=0.82, soften=1)
    _set_band(p, "orange", hue=-4, saturation=10, luminance=2, width=48, feather=0.82, soften=1)
    _set_band(p, "cyan", hue=5, saturation=16, luminance=4, width=55, feather=0.82, soften=1)
    _set_band(p, "green", saturation=-18, luminance=-3, width=60, feather=0.88, soften=1)

    p2 = default_pass("pass-2", "Highlight Control")
    p2["selectedBandId"] = "yellow"
    _set_range(p2, True, 0.55, 1.0, 0.12, "Highlights", boost=False, soften=1)
    _set_band(p2, "yellow", saturation=-5, luminance=-4, width=55, feather=0.85, soften=1)
    _set_band(p2, "magenta", saturation=-8, luminance=-2, width=42, feather=0.85, soften=1)
    recipe["passes"] = [p, p2]
    return recipe


def _preset_reflection_blue_enhance():
    recipe = _preset_base("stars", "Normal", protect_low_sat=True, global_strength=0.8)
    p = default_pass("pass-1", "Reflection Blue Enhance")
    p["selectedBandId"] = "blue"
    _set_range(p, True, 0.12, 0.78, 0.16, "Midtones", boost=True)
    _set_band(p, "blue", saturation=18, luminance=4, width=42, feather=0.82)
    _set_band(p, "cyan", saturation=7, luminance=2, width=40, feather=0.85)
    _set_band(p, "purple", saturation=-6, width=35, feather=0.88)
    recipe["passes"] = [p]
    return recipe


def _preset_magenta_halo_cleanup():
    recipe = _preset_base("stars", "Normal", protect_low_sat=True, global_strength=1.0)
    p = default_pass("pass-1", "Magenta Halo Cleanup")
    p["selectedBandId"] = "magenta"
    _set_range(p, True, 0.45, 1.0, 0.16, "Highlights", boost=False)
    _set_band(p, "magenta", hue=-8, saturation=-22, luminance=-3, width=38, feather=0.86)
    _set_band(p, "purple", hue=-5, saturation=-12, luminance=-2, width=34, feather=0.88)
    recipe["passes"] = [p]
    return recipe


PRESET_BUILDERS = {
    "Galaxy Broadband Color": _preset_galaxy_broadband,
    "HOO Nebula Separation": _preset_hoo_nebula_separation,
    "SHO Palette Refinement": _preset_sho_palette_refinement,
    "Reflection Blue Enhance": _preset_reflection_blue_enhance,
    "Magenta Halo Cleanup": _preset_magenta_halo_cleanup,
}


def preset_names():
    return list(PRESET_BUILDERS.keys())


def preset_recipe(name):
    if name not in PRESET_BUILDERS:
        raise KeyError(f"Unknown Astro Color Mixer preset: {name}")
    return normalize_recipe(PRESET_BUILDERS[name]())

def normalize_recipe(recipe):
    out = deepcopy(recipe or default_recipe())
    out.setdefault("imageType", "stars")
    out.setdefault("sensitivity", "Normal")
    out.setdefault("globalStrength", 1.0)
    out.setdefault("protectionControls", {})
    pc = out["protectionControls"]
    pc["protectStars"] = pc.get("protectStars", True)
    low_sat_value = pc.get("protectLowSaturation", 1.0)
    if isinstance(low_sat_value, (bool, np.bool_)):
        low_sat_value = 1.0 if low_sat_value else 0.0
    try:
        low_sat_value = float(low_sat_value)
    except (TypeError, ValueError):
        low_sat_value = 1.0
    pc["protectLowSaturation"] = float(np.clip(low_sat_value, 0.0, 1.0))
    pc["starMaskStrength"] = normalize_star_mask_strength(pc.get("starMaskStrength", "Strong"))
    ranges = SENSITIVITY_RANGES.get(out["sensitivity"], SENSITIVITY_RANGES["Normal"])
    if not out.get("passes"):
        out["passes"] = [default_pass()]
    for idx, p in enumerate(out["passes"][:4]):
        d = default_pass(p.get("id", f"pass-{idx + 1}"), p.get("name", p.get("label", f"Pass {idx + 1}")))
        d.update(p)
        d["label"] = d.get("name", d.get("label", f"Pass {idx + 1}"))
        band_map = {b["id"]: b for b in d.get("bands", [])}
        bands = []
        for bd in BAND_DEFS:
            source_band = band_map.get(bd["id"], {})
            b = make_default_band(bd)
            b.update(source_band)
            b["hueShift"] = float(np.clip(b.get("hueShift", 0), -ranges["hueShift"], ranges["hueShift"]))
            b["saturation"] = float(np.clip(b.get("saturation", 0), -ranges["saturation"], ranges["saturation"]))
            b["luminance"] = float(np.clip(b.get("luminance", 0), -ranges["luminance"], ranges["luminance"]))
            mode = str(b.get("luminanceMode", "slider"))
            b["luminanceMode"] = "curve" if mode == "curve" else "slider"
            curve = b.get("luminanceCurve", {}) if isinstance(b.get("luminanceCurve", {}), dict) else {}
            source_curve_set = b.get("curveSet") if "curveSet" in source_band else None
            b["curveSet"] = _normalize_curve_set(source_curve_set, curve)
            b["luminanceCurve"] = {"points": b["curveSet"]["L"]["points"]}
            b["width"] = float(np.clip(b.get("width", 45), 5, 120))
            b["feather"] = float(np.clip(b.get("feather", 0.75), 0, 1))
            b["maskSoftenRadius"] = float(np.clip(b.get("maskSoftenRadius", 0), 0, MASK_SOFTEN_CONTROL_MAX))
            cm = b.get("customMask", {}) if isinstance(b.get("customMask", {}), dict) else {}
            cm_type = str(cm.get("type", "Chrominance"))
            if cm_type not in CUSTOM_MASK_TYPES:
                cm_type = "Chrominance"
            b["customMask"] = {
                "startHue": float(np.clip(cm.get("startHue", 300.0), 0.0, 360.0)),
                "endHue": float(np.clip(cm.get("endHue", 60.0), 0.0, 360.0)),
                "strength": float(np.clip(cm.get("strength", 0.75), 0.0, 1.0)),
                "feather": float(np.clip(cm.get("feather", 0.0), 0.0, 1.0)),
                "type": cm_type,
            }
            bands.append(b)
        d["bands"] = bands
        d.setdefault("neutralLuminance", {"luminance": 0, "satStart": 0.04, "satFull": 0.16})
        d["neutralLuminance"]["luminance"] = float(np.clip(d["neutralLuminance"].get("luminance", 0), -ranges["neutral"], ranges["neutral"]))
        d.setdefault("rangeMask", default_pass()["rangeMask"])
        out["passes"][idx] = d
    out["passes"] = out["passes"][:4]
    return out


def _scaled_preview_recipe(recipe, spatial_scale):
    """Scale pixel-radius controls to the dimensions of the preview proxy."""
    scaled = deepcopy(recipe)
    scale = float(np.clip(spatial_scale, EPS, 1.0))
    for pass_state in scaled.get("passes", []):
        for band in pass_state.get("bands", []):
            band["_previewSpatialScale"] = scale
        range_mask = pass_state.get("rangeMask")
        if isinstance(range_mask, dict):
            range_mask["_previewSpatialScale"] = scale
    return scaled


def band_adjustment_active(band):
    curve_active = band.get("luminanceMode") == "curve" and _curve_set_active(band.get("curveSet"))
    return abs(band.get("hueShift", 0)) > EPS or abs(band.get("saturation", 0)) > EPS or abs(band.get("luminance", 0)) > EPS or curve_active


def low_saturation_protection(recipe):
    value = recipe.get("protectionControls", {}).get("protectLowSaturation", 1.0)
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if value else 0.0
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 1.0


def compute_band_mask(source_hsl, band, protection, recipe, range_values=None, star_mask=None):
    pc = recipe["protectionControls"]
    if band.get("custom") or band.get("id") == "custom":
        m = custom_color_mask(source_hsl, band)
    else:
        m = hue_mask(source_hsl["h"], band["center"], band["width"], band["feather"])
    soften = _effective_mask_soften_radius(band.get("maskSoftenRadius", 0) or 0)
    soften *= float(band.get("_previewSpatialScale", 1.0))
    low_sat_protection = low_saturation_protection(recipe)

    # For images with stars, soften the target selection before applying
    # protection gates. This prevents convolution from filling protected
    # star and highlight regions with neighboring mask values.
    if recipe.get("imageType") != "starless" and soften > EPS:
        if range_values is not None:
            m *= range_values
        m = _soften_mask(m.astype(np.float32), soften)
        if low_sat_protection > EPS:
            saturation_gate = smoothstep(protection["satFloor"], protection["satFull"], source_hsl["s"])
            m *= (1.0 - low_sat_protection) + low_sat_protection * saturation_gate
        m *= smoothstep(protection["darkFloor"], protection["darkFull"], source_hsl["l"])
        if pc.get("protectStars", True):
            m *= 1.0 - smoothstep(protection["highlightStart"], protection["highlightFull"], source_hsl["l"])
        if star_mask is not None:
            suppression = star_mask_strength_settings(pc.get("starMaskStrength", "Strong"))["suppress"]
            m *= 1.0 - suppression * star_mask
        m *= float(np.clip(recipe.get("globalStrength", 1.0), 0.0, 1.0))
        return clamp01(m).astype(np.float32)

    if low_sat_protection > EPS:
        saturation_gate = smoothstep(protection["satFloor"], protection["satFull"], source_hsl["s"])
        m *= (1.0 - low_sat_protection) + low_sat_protection * saturation_gate
    m *= smoothstep(protection["darkFloor"], protection["darkFull"], source_hsl["l"])
    if pc.get("protectStars", True):
        m *= 1.0 - smoothstep(protection["highlightStart"], protection["highlightFull"], source_hsl["l"])
    if range_values is not None:
        m *= range_values
    m *= float(np.clip(recipe.get("globalStrength", 1.0), 0.0, 1.0))
    if star_mask is not None:
        m *= 1.0 - star_mask_strength_settings(recipe["protectionControls"].get("starMaskStrength", "Strong"))["suppress"] * star_mask
    if recipe.get("imageType") == "starless" and soften > EPS:
        m = _soften_mask(m.astype(np.float32), soften)
    return clamp01(m).astype(np.float32)


def apply_single_band(working, source_hsl, mask_hsl, band, protection, recipe, range_values=None, star_mask=None):
    mask = compute_band_mask(mask_hsl, band, protection, recipe, range_values, star_mask)
    if np.max(mask) <= 0:
        return working

    active = mask > EPS
    active_fraction = float(np.mean(active))
    use_sparse = active_fraction < 0.65

    sat_adjust = float(band.get("saturation", 0.0)) / 100.0
    hue_shift = float(band.get("hueShift", 0.0))
    lum = float(band.get("luminance", 0.0)) / 100.0
    use_curves = band.get("luminanceMode") == "curve"
    curve_set = _normalize_curve_set(band.get("curveSet"), band.get("luminanceCurve"))
    luminance_lut = None
    rgb_luts = None
    low_sat_reach = 1.0 - low_saturation_protection(recipe)
    if use_curves:
        luminance_lut = _curve_lut(curve_set["L"]["points"])
        rgb_luts = [_curve_lut(curve_set[channel]["points"]) for channel in ("R", "G", "B")]

    if use_sparse:
        output = working.copy()
        pixels = working[active]
        m = mask[active]
        y = (0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]).astype(np.float32)
        chroma = pixels - y[:, None]

        sat = sat_adjust
        if sat_adjust > 0 and low_sat_reach > EPS:
            source_sat = source_hsl["s"][active]
            low_reach = 1.0 - smoothstep(0.02, 0.18, source_sat)
            signal_gate = smoothstep(0.10, 0.34, y) * (1.0 - smoothstep(0.82, 0.98, y))
            sat = sat_adjust * (1.0 + 2.4 * low_sat_reach * low_reach * signal_gate)
        sat_scale = np.maximum(0.0, 1.0 + sat * m)
        chroma = chroma * sat_scale[:, None]

        if abs(hue_shift) > EPS:
            angle = (hue_shift * np.pi / 180.0) * m
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            dot = chroma[:, 0] * AXIS[0] + chroma[:, 1] * AXIS[1] + chroma[:, 2] * AXIS[2]
            cross = np.empty_like(chroma)
            cross[:, 0] = AXIS[1] * chroma[:, 2] - AXIS[2] * chroma[:, 1]
            cross[:, 1] = AXIS[2] * chroma[:, 0] - AXIS[0] * chroma[:, 2]
            cross[:, 2] = AXIS[0] * chroma[:, 1] - AXIS[1] * chroma[:, 0]
            chroma = chroma * cos_a[:, None] + cross * sin_a[:, None] + AXIS[None, :] * dot[:, None] * (1.0 - cos_a[:, None])

        if luminance_lut is not None:
            y_curve = _apply_curve_lut(y, luminance_lut)
            y2 = y + (CURVE_RESPONSE_GAIN * m) * (y_curve - y)
        else:
            y2 = y + (lum * POSITIVE_LUMINANCE_GAIN) * m * (1.0 - y) if lum >= 0 else y + lum * m * y
        band_output = np.clip(y2[:, None] + chroma, 0.0, 1.0).astype(np.float32)
        if rgb_luts is not None:
            rgb_curve = np.empty_like(band_output)
            for channel_index, lut in enumerate(rgb_luts):
                rgb_curve[:, channel_index] = _apply_curve_lut(band_output[:, channel_index], lut)
            band_output = band_output + (CURVE_RESPONSE_GAIN * m)[:, None] * (rgb_curve - band_output)
        output[active] = np.clip(band_output, 0.0, 1.0).astype(np.float32)
        return output.astype(np.float32, copy=False)

    y = luma709(working)
    chroma = working - y[:, :, None]

    if sat_adjust > 0 and low_sat_reach > EPS:
        low_reach = 1.0 - smoothstep(0.02, 0.18, source_hsl["s"])
        signal_gate = smoothstep(0.10, 0.34, y) * (1.0 - smoothstep(0.82, 0.98, y))
        sat_adjust = sat_adjust * (1.0 + 2.4 * low_sat_reach * low_reach * signal_gate)
    sat_scale = np.maximum(0.0, 1.0 + sat_adjust * mask)
    chroma = chroma * sat_scale[:, :, None]

    if abs(hue_shift) > EPS:
        angle = (hue_shift * np.pi / 180.0) * mask
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        dot = chroma[:, :, 0] * AXIS[0] + chroma[:, :, 1] * AXIS[1] + chroma[:, :, 2] * AXIS[2]
        cross = np.empty_like(chroma)
        cross[:, :, 0] = AXIS[1] * chroma[:, :, 2] - AXIS[2] * chroma[:, :, 1]
        cross[:, :, 1] = AXIS[2] * chroma[:, :, 0] - AXIS[0] * chroma[:, :, 2]
        cross[:, :, 2] = AXIS[0] * chroma[:, :, 1] - AXIS[1] * chroma[:, :, 0]
        chroma = chroma * cos_a[:, :, None] + cross * sin_a[:, :, None] + AXIS[None, None, :] * dot[:, :, None] * (1.0 - cos_a[:, :, None])

    if luminance_lut is not None:
        y_curve = _apply_curve_lut(y, luminance_lut)
        y2 = y + (CURVE_RESPONSE_GAIN * mask) * (y_curve - y)
    else:
        y2 = y + (lum * POSITIVE_LUMINANCE_GAIN) * mask * (1.0 - y) if lum >= 0 else y + lum * mask * y
    band_output = clamp01(y2[:, :, None] + chroma).astype(np.float32)
    if rgb_luts is not None:
        rgb_curve = np.empty_like(band_output)
        for channel_index, lut in enumerate(rgb_luts):
            rgb_curve[:, :, channel_index] = _apply_curve_lut(band_output[:, :, channel_index], lut)
        band_output = band_output + (CURVE_RESPONSE_GAIN * mask)[:, :, None] * (rgb_curve - band_output)
    return clamp01(band_output).astype(np.float32)

def apply_neutral_luminance(working, source_hsl, neutral, protection, recipe, range_values=None, star_mask=None):
    lum = float(neutral.get("luminance", 0.0)) / 100.0
    if abs(lum) <= EPS:
        return working
    pc = recipe["protectionControls"]
    neutral_mask = 1.0 - smoothstep(neutral.get("satStart", 0.04), neutral.get("satFull", 0.16), source_hsl["s"])
    dark_floor = protection["darkFloor"] * (0.25 if range_values is not None else 1.0)
    dark_full = protection["darkFull"] * (0.6 if range_values is not None else 1.0)
    mask = neutral_mask * smoothstep(dark_floor, dark_full, source_hsl["l"])
    if pc.get("protectStars", True):
        mask *= 1.0 - smoothstep(protection["highlightStart"], protection["highlightFull"], source_hsl["l"])
    if range_values is not None:
        mask *= range_values
    mask *= float(np.clip(recipe.get("globalStrength", 1.0), 0.0, 1.0))
    if star_mask is not None:
        mask *= 1.0 - star_mask_strength_settings(recipe["protectionControls"].get("starMaskStrength", "Strong"))["suppress"] * star_mask
    y = luma709(working)
    chroma = working - y[:, :, None]
    y2 = y + (lum * POSITIVE_LUMINANCE_GAIN) * mask * (1.0 - y) if lum >= 0 else y + lum * mask * y
    return clamp01(y2[:, :, None] + chroma).astype(np.float32)


def _soften_support(amount):
    """Return the exact pixel support used by a possibly fractional soften radius."""
    amount = float(np.clip(amount, 0.0, MASK_SOFTEN_EFFECTIVE_MAX))
    if amount <= EPS:
        return 0
    return sum(_gaussian_box_radii(int(np.ceil(amount))))


def _pass_tile_halo(pass_state, recipe):
    adjusted = [band for band in pass_state["bands"] if band_adjustment_active(band)]
    range_mask = pass_state.get("rangeMask", {})
    range_support = 0
    if range_mask.get("enabled", False):
        range_soften = _effective_mask_soften_radius(range_mask.get("maskSoftenRadius", 0) or 0)
        range_support = _soften_support(range_soften)
    base_support = max(2 if adjusted else 0, range_support)
    halo = base_support
    for band in adjusted:
        soften = _effective_mask_soften_radius(band.get("maskSoftenRadius", 0) or 0)
        halo = max(halo, base_support + _soften_support(soften))
    return halo


def _tile_regions(height, width, core_size, halo):
    for y0 in range(0, height, core_size):
        y1 = min(height, y0 + core_size)
        for x0 in range(0, width, core_size):
            x1 = min(width, x0 + core_size)
            ey0 = max(0, y0 - halo)
            ey1 = min(height, y1 + halo)
            ex0 = max(0, x0 - halo)
            ex1 = min(width, x1 + halo)
            yield (y0, y1, x0, x1), (ey0, ey1, ex0, ex1)


def _build_star_mask_tiled(source, strength, tile_size=1024, progress=None):
    """Build the shared scalar star mask without full-frame HSL temporaries."""
    settings = star_mask_strength_settings(strength)
    halo = 2 + int(settings["radius"]) + int(settings["blurRadius"])
    height, width = source.shape[:2]
    result = np.empty((height, width), dtype=np.float32)
    regions = list(_tile_regions(height, width, max(64, int(tile_size)), halo))
    report_every = max(1, len(regions) // 10)
    started = time.perf_counter()
    for index, (core, expanded) in enumerate(regions, start=1):
        y0, y1, x0, x1 = core
        ey0, ey1, ex0, ex1 = expanded
        tile_luma = luma709(source[ey0:ey1, ex0:ex1])
        tile_mask = star_protection_mask(tile_luma, strength)
        cy0, cy1 = y0 - ey0, y1 - ey0
        cx0, cx1 = x0 - ex0, x1 - ex0
        result[y0:y1, x0:x1] = tile_mask[cy0:cy1, cx0:cx1]
        if progress is not None and (index == len(regions) or index % report_every == 0):
            progress(f"star protection tiles {index}/{len(regions)}", time.perf_counter() - started)
    return result


def _prepare_tiled_input(data, take_ownership):
    working = _ensure_rgb(np.asarray(data, dtype=np.float32))
    can_reuse = bool(take_ownership and working.flags.writeable)
    if not can_reuse:
        working = working.copy()
    if working.size and float(np.nanmax(working)) > 1.5:
        working /= np.float32(65535.0)
    np.clip(working, 0.0, 1.0, out=working)
    return working


def process_image_tiled(
    data,
    recipe,
    timings=None,
    progress=None,
    tile_size=1024,
    take_ownership=False,
):
    """Process full-resolution data with two RGB buffers and haloed tile scratch."""
    total_start = time.perf_counter()

    def record(label, elapsed):
        if timings is not None:
            timings.append((label, elapsed))
        if progress is not None:
            progress(label, elapsed)

    start = time.perf_counter()
    recipe = normalize_recipe(recipe)
    source = _prepare_tiled_input(data, take_ownership)
    protection = PROTECTION_PRESETS.get(recipe["imageType"], PROTECTION_PRESETS["stars"])
    active_passes = []
    for pass_index, pass_state in enumerate(recipe["passes"], start=1):
        adjusted = [band for band in pass_state["bands"] if band_adjustment_active(band)]
        neutral_active = abs(pass_state["neutralLuminance"].get("luminance", 0.0)) > EPS
        if pass_state.get("enabled", True) and (adjusted or neutral_active):
            active_passes.append((pass_index, pass_state, adjusted, neutral_active))
    destination = np.empty_like(source) if active_passes else None
    shared_star_mask = None
    record("normalize input", time.perf_counter() - start)

    for pass_index, pass_state, adjusted, neutral_active in active_passes:
        pass_start = time.perf_counter()
        pass_label = pass_state.get("label", pass_state.get("name", f"Pass {pass_index}"))
        stage_times = {
            "source HSL": 0.0,
            "mask HSL": 0.0,
            "range mask": 0.0,
            "neutral luminance": 0.0,
        }
        for band in adjusted:
            stage_times[f"apply {band.get('label', band.get('id', 'band'))}"] = 0.0

        use_star_mask = recipe["protectionControls"].get("protectStars", True) and recipe["imageType"] != "starless"
        if use_star_mask:
            start = time.perf_counter()
            if shared_star_mask is None:
                shared_star_mask = _build_star_mask_tiled(
                    source,
                    recipe["protectionControls"].get("starMaskStrength", "Strong"),
                    tile_size=tile_size,
                    progress=progress,
                )
                record(f"{pass_label}: star protection mask", time.perf_counter() - start)
            else:
                record(f"{pass_label}: star protection mask reused", time.perf_counter() - start)
        else:
            record(f"{pass_label}: star protection skipped", 0.0)

        halo = _pass_tile_halo(pass_state, recipe)
        height, width = source.shape[:2]
        regions = list(_tile_regions(height, width, max(64, int(tile_size)), halo))
        report_every = max(1, len(regions) // 10)
        for tile_index, (core, expanded) in enumerate(regions, start=1):
            y0, y1, x0, x1 = core
            ey0, ey1, ex0, ex1 = expanded
            source_tile = source[ey0:ey1, ex0:ex1]
            working_tile = source_tile.copy()

            start = time.perf_counter()
            source_hsl = rgb_to_hsl(source_tile)
            stage_times["source HSL"] += time.perf_counter() - start

            start = time.perf_counter()
            mask_source = _box_blur(source_tile, 2) if adjusted else source_tile
            mask_hsl = rgb_to_hsl(mask_source)
            stage_times["mask HSL"] += time.perf_counter() - start

            start = time.perf_counter()
            range_values = (
                compute_range_mask(source_hsl["y"], pass_state["rangeMask"])
                if pass_state["rangeMask"].get("enabled", False)
                else None
            )
            stage_times["range mask"] += time.perf_counter() - start
            star_tile = shared_star_mask[ey0:ey1, ex0:ex1] if use_star_mask else None

            for band in adjusted:
                label = f"apply {band.get('label', band.get('id', 'band'))}"
                start = time.perf_counter()
                working_tile = apply_single_band(
                    working_tile, source_hsl, mask_hsl, band, protection, recipe, range_values, star_tile
                )
                stage_times[label] += time.perf_counter() - start
            if neutral_active:
                start = time.perf_counter()
                working_tile = apply_neutral_luminance(
                    working_tile,
                    source_hsl,
                    pass_state["neutralLuminance"],
                    protection,
                    recipe,
                    range_values,
                    star_tile,
                )
                stage_times["neutral luminance"] += time.perf_counter() - start

            cy0, cy1 = y0 - ey0, y1 - ey0
            cx0, cx1 = x0 - ex0, x1 - ex0
            destination[y0:y1, x0:x1] = working_tile[cy0:cy1, cx0:cx1]
            if progress is not None and (tile_index == len(regions) or tile_index % report_every == 0):
                progress(f"{pass_label}: tiles {tile_index}/{len(regions)}", time.perf_counter() - pass_start)

        record(f"{pass_label}: source HSL", stage_times["source HSL"])
        record(
            f"{pass_label}: mask HSL" if adjusted else f"{pass_label}: mask HSL skipped",
            stage_times["mask HSL"],
        )
        record(
            f"{pass_label}: range mask" if pass_state["rangeMask"].get("enabled", False) else f"{pass_label}: range mask skipped",
            stage_times["range mask"],
        )
        for band in adjusted:
            label = f"apply {band.get('label', band.get('id', 'band'))}"
            record(f"{pass_label}: {label}", stage_times[label])
        if neutral_active:
            record(f"{pass_label}: neutral luminance", stage_times["neutral luminance"])
        source, destination = destination, source
        record(f"{pass_label}: pass total", time.perf_counter() - pass_start)

    start = time.perf_counter()
    np.clip(source, 0.0, 1.0, out=source)
    record("final clamp", time.perf_counter() - start)
    record("total processing", time.perf_counter() - total_start)
    return source


def process_image(data, recipe, timings=None, progress=None):
    total_start = time.perf_counter()

    def mark(label, start):
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings.append((label, elapsed))
        if progress is not None:
            progress(label, elapsed)
    start = time.perf_counter()
    recipe = normalize_recipe(recipe)
    working = clamp01(_ensure_rgb(_normalize_range(data))).astype(np.float32, copy=True)
    protection = PROTECTION_PRESETS.get(recipe["imageType"], PROTECTION_PRESETS["stars"])
    shared_star_mask = None
    mark("normalize input", start)

    for pass_index, p in enumerate(recipe["passes"], start=1):
        pass_start = time.perf_counter()
        pass_label = p.get("label", p.get("name", f"Pass {pass_index}"))
        if not p.get("enabled", True):
            continue
        adjusted = [b for b in p["bands"] if band_adjustment_active(b)]
        neutral_active = abs(p["neutralLuminance"].get("luminance", 0.0)) > EPS
        if not adjusted and not neutral_active:
            continue

        start = time.perf_counter()
        source_hsl = rgb_to_hsl(working)
        mark(f"{pass_label}: source HSL", start)

        start = time.perf_counter()
        mask_source = _box_blur(working, 2) if adjusted else working
        mask_hsl = rgb_to_hsl(mask_source)
        mark(f"{pass_label}: mask HSL" if adjusted else f"{pass_label}: mask HSL skipped", start)

        star_mask = None
        if recipe["protectionControls"].get("protectStars", True) and recipe["imageType"] != "starless":
            start = time.perf_counter()
            if shared_star_mask is None:
                shared_star_mask = star_protection_mask(source_hsl["y"], recipe["protectionControls"].get("starMaskStrength", "Strong"))
                mark(f"{pass_label}: star protection mask", start)
            else:
                mark(f"{pass_label}: star protection mask reused", start)
            star_mask = shared_star_mask
        else:
            if timings is not None:
                timings.append((f"{pass_label}: star protection skipped", 0.0))
            if progress is not None:
                progress(f"{pass_label}: star protection skipped", 0.0)
        start = time.perf_counter()
        range_values = compute_range_mask(source_hsl["y"], p["rangeMask"]) if p["rangeMask"].get("enabled", False) else None
        mark(f"{pass_label}: range mask" if range_values is not None else f"{pass_label}: range mask skipped", start)

        for b in adjusted:
            start = time.perf_counter()
            working = apply_single_band(working, source_hsl, mask_hsl, b, protection, recipe, range_values, star_mask)
            mark(f"{pass_label}: apply {b.get('label', b.get('id', 'band'))}", start)
        if neutral_active:
            start = time.perf_counter()
            working = apply_neutral_luminance(working, source_hsl, p["neutralLuminance"], protection, recipe, range_values, star_mask)
            mark(f"{pass_label}: neutral luminance", start)
        mark(f"{pass_label}: pass total", pass_start)

    start = time.perf_counter()
    output = clamp01(working).astype(np.float32)
    mark("final clamp", start)
    mark("total processing", total_start)
    return output

def compute_preview_mask(data, recipe, pass_index, band_id, mode):
    recipe = normalize_recipe(recipe)
    rgb = clamp01(_ensure_rgb(_normalize_range(data))).astype(np.float32)
    p = recipe["passes"][pass_index]
    hsl = rgb_to_hsl(rgb)
    protection = PROTECTION_PRESETS.get(recipe["imageType"], PROTECTION_PRESETS["stars"])
    star = None
    if recipe["protectionControls"].get("protectStars", True) and recipe["imageType"] != "starless":
        star = star_protection_mask(hsl["y"], recipe["protectionControls"].get("starMaskStrength", "Strong"))
    range_values = compute_range_mask(hsl["y"], p["rangeMask"])
    band = next((b for b in p["bands"] if b["id"] == band_id), p["bands"][0])
    mask_hsl = rgb_to_hsl(_box_blur(rgb, 2))
    if mode == "bandMask":
        values = compute_band_mask(mask_hsl, band, protection, recipe, None, star)
    elif mode == "rangeMask":
        values = range_values
    elif mode == "starMask":
        values = star if star is not None else np.zeros_like(hsl["y"])
    else:
        values = compute_band_mask(
            mask_hsl,
            band,
            protection,
            recipe,
            range_values if p["rangeMask"].get("enabled", False) else None,
            star,
        )
    return np.repeat(clamp01(values)[:, :, None], 3, axis=2).astype(np.float32)


# ======================================================================
# GUI Widgets
# ======================================================================

class ArrowDoubleSpinBox(QDoubleSpinBox):
    """Restore native arrow indicators when QSS owns the button geometry."""

    def paintEvent(self, event):
        super().paintEvent(event)
        spin_option = QStyleOptionSpinBox()
        self.initStyleOption(spin_option)
        painter = QPainter(self)
        for subcontrol, primitive in (
            (QStyle.SubControl.SC_SpinBoxUp, QStyle.PrimitiveElement.PE_IndicatorArrowUp),
            (QStyle.SubControl.SC_SpinBoxDown, QStyle.PrimitiveElement.PE_IndicatorArrowDown),
        ):
            rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox,
                spin_option,
                subcontrol,
                self,
            )
            if rect.isValid() and not rect.isEmpty():
                arrow_option = QStyleOption()
                arrow_option.initFrom(self)
                arrow_option.rect = rect.adjusted(4, 2, -4, -2)
                self.style().drawPrimitive(primitive, arrow_option, painter, self)

def _make_slider_row(parent_layout, label_text, minv, maxv, default, decimals=2, on_change=None, extra_widget=None):
    row = QHBoxLayout()
    label = QLabel(label_text)
    label.setMinimumWidth(190)
    label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
    spin = ArrowDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minv, maxv)
    spin.setSingleStep(max((maxv - minv) / 100.0, 1.0 / (10 ** decimals)))
    spin.setValue(default)
    spin.setMinimumWidth(82)
    spin.setMinimumHeight(26)
    scale = 10 ** decimals
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(int(round(minv * scale)), int(round(maxv * scale)))
    slider.setValue(int(round(default * scale)))

    def slider_to_spin(v):
        spin.blockSignals(True)
        spin.setValue(v / scale)
        spin.blockSignals(False)

    def spin_to_slider(v):
        slider.blockSignals(True)
        slider.setValue(int(round(v * scale)))
        slider.blockSignals(False)

    slider.valueChanged.connect(slider_to_spin)
    spin.valueChanged.connect(spin_to_slider)
    if on_change:
        slider.valueChanged.connect(on_change)
        spin.valueChanged.connect(on_change)
    row.addWidget(label)
    row.addWidget(spin)
    row_widgets = [label, spin, slider]
    if extra_widget is not None:
        row.addWidget(extra_widget)
        row_widgets.insert(2, extra_widget)
    row.addWidget(slider)
    spin._acm_slider = slider
    spin._acm_extra_widget = extra_widget
    spin._acm_row_widgets = tuple(row_widgets)
    parent_layout.addLayout(row)
    return spin, slider


class HueRangeWheel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_hue = 300.0
        self.end_hue = 60.0
        self.setMinimumHeight(112)
        self.setMaximumHeight(132)
        self.setToolTip("Selected custom hue range")

    def set_range(self, start_hue, end_hue):
        self.start_hue = float(start_hue) % 360.0
        self.end_hue = float(end_hue) % 360.0
        self.update()

    def _angle_point(self, cx, cy, radius, hue):
        theta = np.deg2rad(float(hue) - 90.0)
        return int(round(cx + np.cos(theta) * radius)), int(round(cy + np.sin(theta) * radius))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(37, 37, 37))

        size = min(self.width() - 12, self.height() - 12)
        if size <= 8:
            return
        left = int((self.width() - size) / 2)
        top = int((self.height() - size) / 2)
        cx = left + size / 2.0
        cy = top + size / 2.0
        span = (self.end_hue - self.start_hue) % 360.0
        if span <= EPS:
            span = 360.0

        painter.setPen(Qt.PenStyle.NoPen)
        for hue in range(360):
            color = QColor.fromHsv(hue, 210, 210)
            color.setAlpha(72)
            painter.setBrush(color)
            painter.drawPie(left, top, size, size, int((90 - hue) * 16), -18)

        for step in range(int(round(span)) + 1):
            hue = (self.start_hue + step) % 360.0
            color = QColor.fromHsv(int(round(hue)) % 360, 255, 255)
            color.setAlpha(245)
            painter.setBrush(color)
            painter.drawPie(left, top, size, size, int((90 - hue) * 16), -18)

        inner = int(size * 0.54)
        inner_left = int(round(cx - inner / 2.0))
        inner_top = int(round(cy - inner / 2.0))
        painter.setBrush(QColor(37, 37, 37))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(inner_left, inner_top, inner, inner)

        radius = size * 0.49
        inner_radius = size * 0.28
        for hue, color in ((self.start_hue, QColor(255, 255, 255)), (self.end_hue, QColor(20, 20, 20))):
            x0, y0 = self._angle_point(cx, cy, inner_radius, hue)
            x1, y1 = self._angle_point(cx, cy, radius, hue)
            painter.setPen(QPen(color, 3))
            painter.drawLine(x0, y0, x1, y1)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawEllipse(left, top, size, size)
        painter.drawEllipse(inner_left, inner_top, inner, inner)


class CurvesEditor(QWidget):
    curveChanged = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setMouseTracking(True)
        self.curve_set = _identity_curve_set()
        self.active_channel = "L"
        self.drag_idx = -1
        self.hover_idx = -1
        self.setToolTip("Left-click to add or drag points. Right-click an interior point to remove it.")

    @property
    def points(self):
        return self.curve_set[self.active_channel]["points"]

    @points.setter
    def points(self, value):
        self.curve_set[self.active_channel]["points"] = _clean_curve_points(value)

    def load_curve_set(self, curve_set):
        self.curve_set = _normalize_curve_set(curve_set)
        self.update()

    def set_active_channel(self, channel):
        if channel not in CURVE_CHANNELS:
            return
        self.active_channel = channel
        self.drag_idx = -1
        self.hover_idx = -1
        self.update()

    def reset_active(self):
        self.curve_set[self.active_channel] = {"points": _identity_curve_points()}
        self.update()
        self.curveChanged.emit(self.curve_set)

    def reset_all(self):
        self.curve_set = _identity_curve_set()
        self.update()
        self.curveChanged.emit(self.curve_set)

    def _plot_rect(self):
        margin = 16
        return margin, margin, max(1, self.width() - 2 * margin), max(1, self.height() - 2 * margin)

    def _to_screen(self, x, y):
        left, top, width, height = self._plot_rect()
        return left + x * width, top + (1.0 - y) * height

    def _from_screen(self, pos):
        left, top, width, height = self._plot_rect()
        x = np.clip((pos.x() - left) / width, 0.0, 1.0)
        y = np.clip(1.0 - ((pos.y() - top) / height), 0.0, 1.0)
        return float(x), float(y)

    def _hit_point(self, pos):
        for i, (x, y) in enumerate(self.points):
            sx, sy = self._to_screen(x, y)
            if (sx - pos.x()) ** 2 + (sy - pos.y()) ** 2 <= 9 ** 2:
                return i
        return -1

    def _draw_curve(self, painter, channel, active):
        left, top, width, height = self._plot_rect()
        lut = _curve_lut(self.curve_set[channel]["points"], size=256)
        path = QPainterPath()
        path.moveTo(left, top + height - float(lut[0]) * height)
        for i, value in enumerate(lut):
            x = left + (i / 255.0) * width
            y = top + height - float(value) * height
            path.lineTo(x, y)
        color = QColor(CURVE_CHANNEL_COLORS[channel])
        color.setAlpha(255 if active else 120)
        painter.setPen(QPen(color, 2 if active else 1))
        painter.drawPath(path)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        left, top, width, height = self._plot_rect()
        painter.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            x = int(left + width * i / 4)
            y = int(top + height * i / 4)
            painter.drawLine(x, top, x, top + height)
            painter.drawLine(left, y, left + width, y)
        painter.setPen(QPen(QColor(92, 92, 92), 1))
        painter.drawRect(left, top, width, height)
        painter.setPen(QPen(QColor(95, 95, 95), 1))
        painter.drawLine(left, top + height, left + width, top)

        for channel in CURVE_CHANNELS:
            if channel != self.active_channel:
                self._draw_curve(painter, channel, False)
        self._draw_curve(painter, self.active_channel, True)

        active_color = QColor(CURVE_CHANNEL_COLORS[self.active_channel])
        for i, (x, y) in enumerate(self.points):
            sx, sy = self._to_screen(x, y)
            radius = 6 if i in (self.drag_idx, self.hover_idx) else 5
            painter.setBrush(QColor(32, 32, 32))
            painter.setPen(QPen(active_color, 2))
            painter.drawEllipse(int(sx - radius), int(sy - radius), radius * 2, radius * 2)

    def mousePressEvent(self, event):
        idx = self._hit_point(event.pos())
        if event.button() == Qt.MouseButton.RightButton:
            if 0 < idx < len(self.points) - 1:
                points = list(self.points)
                points.pop(idx)
                self.points = points
                self.hover_idx = -1
                self.update()
                self.curveChanged.emit(self.curve_set)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if idx >= 0:
            self.drag_idx = idx
            return
        x, y = self._from_screen(event.pos())
        points = list(self.points)
        points.append([x, y])
        self.points = points
        self.drag_idx = min(range(len(self.points)), key=lambda i: abs(self.points[i][0] - x))
        self.update()
        self.curveChanged.emit(self.curve_set)

    def mouseMoveEvent(self, event):
        x, y = self._from_screen(event.pos())
        if self.drag_idx >= 0:
            idx = self.drag_idx
            points = [list(p) for p in self.points]
            if idx == 0:
                x = 0.0
            elif idx == len(points) - 1:
                x = 1.0
            else:
                x = float(np.clip(x, points[idx - 1][0] + 0.005, points[idx + 1][0] - 0.005))
            points[idx] = [x, y]
            self.points = points
            self.update()
            self.curveChanged.emit(self.curve_set)
            return
        self.hover_idx = self._hit_point(event.pos())
        self.update()

    def mouseReleaseEvent(self, _event):
        self.drag_idx = -1
        self.points = self.points
        self.update()
        self.curveChanged.emit(self.curve_set)


class CurvesDialog(QDialog):
    curveChanged = pyqtSignal(object)

    def __init__(self, band_label, curve_set, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setWindowTitle(f"Curves - {band_label}")
        self.use_slider_requested = False
        layout = QVBoxLayout(self)
        hint = QLabel("Curves apply inside this band's mask. Pick a channel to edit; all curves are shown together.")
        hint.setStyleSheet("color: #c8c8c8;")
        layout.addWidget(hint)

        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Edit channel"))
        self.channel_combo = QComboBox()
        for channel in CURVE_CHANNELS:
            self.channel_combo.addItem(CURVE_CHANNEL_LABELS[channel], channel)
        channel_row.addWidget(self.channel_combo)
        channel_row.addStretch()
        layout.addLayout(channel_row)

        self.editor = CurvesEditor(self)
        self.editor.load_curve_set(curve_set)
        self.editor.curveChanged.connect(self.curveChanged.emit)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        layout.addWidget(self.editor, 1)

        row = QHBoxLayout()
        reset_btn = QPushButton("Reset Channel")
        reset_btn.clicked.connect(self.editor.reset_active)
        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.clicked.connect(self.editor.reset_all)
        use_slider_btn = QPushButton("Use Slider")
        use_slider_btn.setToolTip("Disable these curves and return to the normal luminance slider for this band.")
        use_slider_btn.clicked.connect(self._use_slider)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(reset_btn)
        row.addWidget(reset_all_btn)
        row.addWidget(use_slider_btn)
        row.addStretch()
        row.addWidget(buttons)
        layout.addLayout(row)
        self.resize(470, 390)

    def _on_channel_changed(self):
        self.editor.set_active_channel(self.channel_combo.currentData())

    def _use_slider(self):
        self.use_slider_requested = True
        self.accept()

    def curve_set(self):
        return _normalize_curve_set(self.editor.curve_set)


class ZoomPanScrollArea(QScrollArea):
    def __init__(self, on_wheel_zoom, parent=None):
        super().__init__(parent)
        self._on_wheel_zoom = on_wheel_zoom
        self._panning = False
        self._pan_start = None
        self._scroll_start = (0, 0)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta:
            self._on_wheel_zoom(delta)
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self._scroll_start = (self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(self._scroll_start[0] - (pos.x() - self._pan_start.x()))
            self.verticalScrollBar().setValue(self._scroll_start[1] - (pos.y() - self._pan_start.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class AstroColorMixerApplyWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, raw, recipe):
        super().__init__()
        self.raw = raw
        self.recipe = recipe

    def _emit_progress(self, label, seconds):
        self.progress.emit(f"{label}: {seconds:.3f}s")

    def run(self):
        try:
            timings = []
            start = time.perf_counter()
            data, layout = _to_hwc(self.raw)
            data = _ensure_rgb(np.asarray(data, dtype=np.float32))
            self.raw = None
            elapsed = time.perf_counter() - start
            timings.append(("worker prepare input", elapsed))
            self.progress.emit(f"worker prepare input: {elapsed:.3f}s")
            result = process_image_tiled(
                data,
                self.recipe,
                timings=timings,
                progress=self._emit_progress,
                take_ownership=True,
            )
            data = None
            start = time.perf_counter()
            pixeldata = _from_hwc(result, layout)
            elapsed = time.perf_counter() - start
            timings.append(("worker restore layout", elapsed))
            self.progress.emit(f"worker restore layout: {elapsed:.3f}s")
            self.finished.emit({"pixeldata": pixeldata, "timings": timings})
        except Exception as e:
            self.failed.emit(str(e))


class AstroColorMixerWindow(QMainWindow):
    PREVIEW_MAX_DIM = 720

    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.recipe = default_recipe()
        self.pass_index = 0
        self.proxy_data = None
        self.preview_spatial_scale = 1.0
        self.preview_display_range = None
        self.original_pixmap_base = None
        self.processed_pixmap_base = None
        self.preview_zoom = 1.0
        self.show_original_active = False
        self.apply_thread = None
        self.apply_worker = None
        self.close_when_finished = False
        self.curves_dialog = None
        self.band_controls = {"hueShift": {}, "saturation": {}, "luminance": {}, "width": {}, "feather": {}, "soften": {}, "customStart": {}, "customEnd": {}, "customStrength": {}, "customFeather": {}, "customType": {}, "customWheel": {}}
        self.luminance_curve_buttons = {}
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._update_preview)
        self.setWindowTitle(f"Astro Color Mixer for Siril - v{VERSION}")
        self._build_ui()
        self._sync_from_recipe()
        self._refresh_preview_source()

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #303030; color: #f0f0f0; }
            QGroupBox { border: 1px solid #626262; border-radius: 4px; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #ffc43a; }
            QPushButton { background: #454545; border: 1px solid #686868; padding: 5px 9px; border-radius: 3px; }
            QPushButton:hover { background: #505050; }
            QComboBox, QLineEdit { background: #202020; border: 1px solid #555; padding: 3px; }
            QDoubleSpinBox { background: #202020; border: 1px solid #555; padding: 3px 22px 3px 3px; }
            QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555;
                border-bottom: 1px solid #555;
            }
            QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border-left: 1px solid #555;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #505050; }
            QTabWidget::pane { border: 1px solid #555; }
            QTabBar::tab { background: #3c3c3c; padding: 5px 10px; }
            QTabBar::tab:selected { background: #ffc43a; color: #111; }
        """)
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(520)
        controls = QWidget()
        controls_scroll.setWidget(controls)
        left = QVBoxLayout(controls)

        target_box = QGroupBox("Target and Protection")
        tl = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Image type"))
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(["Stars Present", "Starless"])
        self.image_type_combo.currentIndexChanged.connect(self._on_global_changed)
        row.addWidget(self.image_type_combo)
        row.addWidget(QLabel("Sensitivity"))
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["Fine", "Normal", "Advanced", "Strong"])
        self.sensitivity_combo.currentTextChanged.connect(self._on_sensitivity_changed)
        row.addWidget(self.sensitivity_combo)
        tl.addLayout(row)
        self.global_strength_spin, _ = _make_slider_row(tl, "Global Strength", 0.0, 1.0, 1.0, on_change=self._on_global_changed)
        self.protect_stars_check = QCheckBox("Protect compact stars and highlights")
        self.protect_stars_check.setChecked(True)
        self.protect_stars_check.stateChanged.connect(self._on_global_changed)
        tl.addWidget(self.protect_stars_check)
        self.low_sat_protection_spin, low_sat_slider = _make_slider_row(
            tl,
            "Low-Saturation Protection",
            0.0,
            1.0,
            1.0,
            decimals=2,
            on_change=self._on_global_changed,
        )
        low_sat_tooltip = "0 allows adjustments into neutral colors; 1 matches the former checked protection."
        self.low_sat_protection_spin.setToolTip(low_sat_tooltip)
        low_sat_slider.setToolTip(low_sat_tooltip)
        star_strength_row = QHBoxLayout()
        star_strength_label = QLabel("Star Protection")
        star_strength_label.setMinimumWidth(190)
        star_strength_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        self.star_strength_combo = QComboBox()
        self.star_strength_combo.addItems(list(STAR_MASK_STRENGTHS))
        self.star_strength_combo.setCurrentText("Strong")
        self.star_strength_combo.setToolTip("Controls compact-star detection sensitivity, protection radius, and suppression.")
        self.star_strength_combo.currentTextChanged.connect(self._on_global_changed)
        star_strength_row.addWidget(star_strength_label)
        star_strength_row.addWidget(self.star_strength_combo, 1)
        tl.addLayout(star_strength_row)
        target_box.setLayout(tl)
        left.addWidget(target_box)

        pass_box = QGroupBox("Passes")
        pl = QVBoxLayout()
        row = QHBoxLayout()
        self.pass_combo = QComboBox()
        self.pass_combo.currentIndexChanged.connect(self._on_pass_selected)
        row.addWidget(self.pass_combo, 1)
        self.pass_enabled_check = QCheckBox("Enabled")
        self.pass_enabled_check.stateChanged.connect(self._on_pass_enabled_changed)
        row.addWidget(self.pass_enabled_check)
        pl.addLayout(row)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.pass_name_edit = QLineEdit()
        self.pass_name_edit.editingFinished.connect(self._on_pass_name_changed)
        name_row.addWidget(self.pass_name_edit)
        pl.addLayout(name_row)
        btns = QHBoxLayout()
        self.new_pass_btn = QPushButton("New")
        self.dup_pass_btn = QPushButton("Duplicate")
        self.del_pass_btn = QPushButton("Delete")
        self.new_pass_btn.clicked.connect(self._new_pass)
        self.dup_pass_btn.clicked.connect(self._duplicate_pass)
        self.del_pass_btn.clicked.connect(self._delete_pass)
        btns.addWidget(self.new_pass_btn)
        btns.addWidget(self.dup_pass_btn)
        btns.addWidget(self.del_pass_btn)
        pl.addLayout(btns)
        pass_box.setLayout(pl)
        left.addWidget(pass_box)

        band_box = QGroupBox("Color Mixer")
        bl = QVBoxLayout()
        bl.setSpacing(6)
        select_row = QHBoxLayout()
        select_row.addWidget(QLabel("Selected band for masks"))
        self.selected_band_combo = QComboBox()
        for bd in BAND_DEFS:
            self.selected_band_combo.addItem(bd["label"], bd["id"])
        self.selected_band_combo.currentIndexChanged.connect(self._on_selected_band_changed)
        select_row.addWidget(self.selected_band_combo)
        bl.addLayout(select_row)
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(318)
        self.tabs.setMaximumHeight(342)
        default_ranges = SENSITIVITY_RANGES["Normal"]
        for key, title in [("hueShift", "Hue"), ("saturation", "Saturation"), ("luminance", "Luminance")]:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            limit = default_ranges[key]
            for bd in BAND_DEFS:
                extra = None
                if key == "luminance":
                    extra = QToolButton()
                    extra.setText("C")
                    extra.setCheckable(True)
                    extra.setFixedWidth(28)
                    extra.setToolTip("Use/edit L/R/G/B curves for this color band")
                    extra.clicked.connect(lambda _checked=False, bid=bd["id"]: self._edit_luminance_curve(bid))
                    self.luminance_curve_buttons[bd["id"]] = extra
                spin, _ = _make_slider_row(layout, bd["label"], -limit, limit, 0, decimals=2, on_change=self._on_band_control_changed, extra_widget=extra)
                self.band_controls[key][bd["id"]] = spin
            layout.addStretch()
            self.tabs.addTab(tab, title)
        band_shape_scroll = QScrollArea()
        band_shape_scroll.setWidgetResizable(True)
        band_shape_scroll.setFrameShape(QFrame.Shape.NoFrame)
        band_shape = QWidget()
        band_shape_scroll.setWidget(band_shape)
        shape_layout = QVBoxLayout(band_shape)
        shape_layout.setContentsMargins(8, 8, 8, 8)
        shape_layout.setSpacing(6)
        shape_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for bd in BAND_DEFS:
            row_box = QGroupBox(bd["label"])
            row_l = QVBoxLayout()
            row_l.setSpacing(4)
            if bd.get("custom"):
                start_spin, start_slider = _make_slider_row(row_l, "Start Hue", 0.0, 360.0, 300.0, decimals=1, on_change=self._on_band_control_changed)
                end_spin, end_slider = _make_slider_row(row_l, "End Hue", 0.0, 360.0, 60.0, decimals=1, on_change=self._on_band_control_changed)
                strength_spin, _ = _make_slider_row(row_l, "Mask Strength", 0.0, 1.0, 0.75, decimals=2, on_change=self._on_band_control_changed)
                feather_spin, _ = _make_slider_row(row_l, "Feather", 0.0, 1.0, 0.0, decimals=2, on_change=self._on_band_control_changed)
                wheel = HueRangeWheel()
                wheel.set_range(start_spin.value(), end_spin.value())

                def update_custom_wheel(_value=None, w=wheel, s=start_spin, e=end_spin):
                    w.set_range(s.value(), e.value())

                start_spin.valueChanged.connect(update_custom_wheel)
                end_spin.valueChanged.connect(update_custom_wheel)
                start_slider.valueChanged.connect(update_custom_wheel)
                end_slider.valueChanged.connect(update_custom_wheel)
                row_l.addWidget(wheel)
                type_row = QHBoxLayout()
                type_label = QLabel("Mask Type")
                type_label.setMinimumWidth(190)
                type_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
                type_combo = QComboBox()
                type_combo.addItems(list(CUSTOM_MASK_TYPES))
                type_combo.currentTextChanged.connect(self._on_band_control_changed)
                type_row.addWidget(type_label)
                type_row.addWidget(type_combo)
                row_l.addLayout(type_row)
                s_spin, _ = _make_slider_row(row_l, "Soften Radius", 0.0, MASK_SOFTEN_CONTROL_MAX, 0.0, decimals=2, on_change=self._on_band_control_changed)
                self.band_controls["customStart"][bd["id"]] = start_spin
                self.band_controls["customEnd"][bd["id"]] = end_spin
                self.band_controls["customStrength"][bd["id"]] = strength_spin
                self.band_controls["customFeather"][bd["id"]] = feather_spin
                self.band_controls["customType"][bd["id"]] = type_combo
                self.band_controls["customWheel"][bd["id"]] = wheel
                self.band_controls["width"][bd["id"]] = start_spin
                self.band_controls["feather"][bd["id"]] = feather_spin
                self.band_controls["soften"][bd["id"]] = s_spin
            else:
                w_spin, _ = _make_slider_row(row_l, "Hue Radius", 5.0, 120.0, 45.0, decimals=1, on_change=self._on_band_control_changed)
                f_spin, _ = _make_slider_row(row_l, "Feather", 0.0, 1.0, 0.75, on_change=self._on_band_control_changed)
                s_spin, _ = _make_slider_row(row_l, "Mask Soften", 0.0, MASK_SOFTEN_CONTROL_MAX, 0.0, decimals=2, on_change=self._on_band_control_changed)
                self.band_controls["width"][bd["id"]] = w_spin
                self.band_controls["feather"][bd["id"]] = f_spin
                self.band_controls["soften"][bd["id"]] = s_spin
            row_box.setLayout(row_l)
            shape_layout.addWidget(row_box)
        shape_layout.addStretch()
        self.tabs.addTab(band_shape_scroll, "Masks")
        bl.addWidget(self.tabs)
        self.neutral_spin, _ = _make_slider_row(bl, "Neutral Luminance", -20.0, 20.0, 0.0, on_change=self._on_neutral_changed)
        band_box.setLayout(bl)
        left.addWidget(band_box)

        range_box = QGroupBox("Range Mask")
        rl = QVBoxLayout()
        self.range_enabled_check = QCheckBox("Enable Range Mask on active pass")
        self.range_enabled_check.stateChanged.connect(self._on_range_changed)
        rl.addWidget(self.range_enabled_check)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.range_preset_combo = QComboBox()
        self.range_preset_combo.addItems(list(RANGE_PRESETS.keys()) + ["Custom"])
        self.range_preset_combo.currentTextChanged.connect(self._on_range_preset_changed)
        row.addWidget(self.range_preset_combo)
        self.range_boost_check = QCheckBox("Boost")
        self.range_boost_check.stateChanged.connect(self._on_range_changed)
        row.addWidget(self.range_boost_check)
        rl.addLayout(row)
        self.range_low_spin, _ = _make_slider_row(rl, "Low", 0.0, 1.0, 0.0, on_change=self._on_range_changed)
        self.range_high_spin, _ = _make_slider_row(rl, "High", 0.0, 1.0, 1.0, on_change=self._on_range_changed)
        self.range_feather_spin, _ = _make_slider_row(rl, "Feather", 0.0, 0.5, 0.10, on_change=self._on_range_changed)
        self.range_soften_spin, _ = _make_slider_row(rl, "Mask Soften", 0.0, MASK_SOFTEN_CONTROL_MAX, 0.0, decimals=2, on_change=self._on_range_changed)
        range_box.setLayout(rl)
        left.addWidget(range_box)

        file_box = QGroupBox("Adjustment Set")
        fl = QVBoxLayout()
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(preset_names())
        preset_row.addWidget(self.preset_combo, 1)
        self.apply_preset_btn = QPushButton("Apply Preset")
        self.apply_preset_btn.clicked.connect(self._apply_builtin_preset)
        preset_row.addWidget(self.apply_preset_btn)
        fl.addLayout(preset_row)
        file_row = QHBoxLayout()
        self.load_btn = QPushButton("Load JSON")
        self.save_btn = QPushButton("Save JSON")
        self.reset_btn = QPushButton("Reset")
        self.load_btn.clicked.connect(self._load_recipe)
        self.save_btn.clicked.connect(self._save_recipe)
        self.reset_btn.clicked.connect(self._reset_recipe)
        file_row.addWidget(self.load_btn)
        file_row.addWidget(self.save_btn)
        file_row.addWidget(self.reset_btn)
        fl.addLayout(file_row)
        file_box.setLayout(fl)
        left.addWidget(file_box)

        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply to Siril")
        self.apply_btn.clicked.connect(self._apply_to_siril)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        apply_row.addWidget(self.apply_btn)
        apply_row.addStretch()
        apply_row.addWidget(self.close_btn)
        left.addLayout(apply_row)
        left.addStretch()
        root.addWidget(controls_scroll)

        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        top.addWidget(QLabel("Preview"))
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Adjusted", "Original", "Band Mask", "Range Mask", "Combined Mask", "Star Protection Mask"])
        self.preview_mode_combo.currentIndexChanged.connect(self._schedule_preview_update)
        top.addWidget(self.preview_mode_combo)
        self.mask_overlay_check = QCheckBox("Overlay")
        self.mask_overlay_check.setChecked(True)
        self.mask_overlay_check.stateChanged.connect(self._schedule_preview_update)
        top.addWidget(self.mask_overlay_check)
        top.addWidget(QLabel("Opacity"))
        self.mask_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(55)
        self.mask_opacity_slider.setFixedWidth(120)
        self.mask_opacity_slider.valueChanged.connect(self._schedule_preview_update)
        top.addWidget(self.mask_opacity_slider)
        top.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #cfcfcf;")
        top.addWidget(self.status_label)
        rv.addLayout(top)
        self.mask_stats_label = QLabel("")
        self.mask_stats_label.setStyleSheet("color: #b8b8b8; font-size: 10px;")
        self.mask_stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        rv.addWidget(self.mask_stats_label)
        self.preview_label = QLabel("Loading preview...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(520, 520)
        self.preview_label.setStyleSheet("background: #101010; color: #aaa;")
        self.preview_scroll = ZoomPanScrollArea(self._on_wheel_zoom)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rv.addWidget(self.preview_scroll, 1)
        zoom = QHBoxLayout()
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedWidth(34)
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(56)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(34)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_reset_btn = QPushButton("Reset zoom")
        self.zoom_reset_btn.clicked.connect(self._zoom_reset)
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setToolTip("Fit the image to the current preview area")
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        self.refresh_btn = QPushButton("Refresh from Siril")
        self.refresh_btn.clicked.connect(self._refresh_preview_source)
        self.show_original_btn = QPushButton("Show original (hold)")
        self.show_original_btn.pressed.connect(self._show_original_pressed)
        self.show_original_btn.released.connect(self._show_original_released)
        zoom.addWidget(self.zoom_out_btn)
        zoom.addWidget(self.zoom_label)
        zoom.addWidget(self.zoom_in_btn)
        zoom.addWidget(self.zoom_reset_btn)
        zoom.addWidget(self.zoom_fit_btn)
        zoom.addStretch()
        zoom.addWidget(self.refresh_btn)
        zoom.addWidget(self.show_original_btn)
        rv.addLayout(zoom)
        attribution = QLabel("Based on Astro Color Mixer for PixInsight by Patrick Cosgrove")
        attribution.setAlignment(Qt.AlignmentFlag.AlignRight)
        attribution.setStyleSheet("color: #9a9a9a; font-size: 10px;")
        rv.addWidget(attribution)
        root.addWidget(right, 1)
        self.resize(1320, 860)

    def _active_pass(self):
        return self.recipe["passes"][self.pass_index]

    def _active_band(self, band_id):
        return _band(self._active_pass(), band_id)

    def _set_luminance_curve_control_state(self, band):
        bid = band["id"]
        curve_enabled = band.get("luminanceMode") == "curve"
        spin = self.band_controls["luminance"][bid]
        spin.setEnabled(not curve_enabled)
        slider = getattr(spin, "_acm_slider", None)
        if slider is not None:
            slider.setEnabled(not curve_enabled)
        btn = self.luminance_curve_buttons.get(bid)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(curve_enabled)
            btn.setText("C*" if curve_enabled else "C")
            btn.setToolTip("Edit L/R/G/B curves for this color band" if curve_enabled else "Use/edit L/R/G/B curves for this color band")
            btn.blockSignals(False)

    def _edit_luminance_curve(self, band_id):
        if getattr(self, "syncing", False):
            return
        if self.curves_dialog is not None:
            self.curves_dialog.show()
            self.curves_dialog.raise_()
            self.curves_dialog.activateWindow()
            self._set_luminance_curve_control_state(self._active_band(band_id))
            return
        self._write_active_pass_from_controls()
        band = self._active_band(band_id)
        original_mode = band.get("luminanceMode", "slider")
        original_curve_set = deepcopy(_normalize_curve_set(band.get("curveSet"), band.get("luminanceCurve")))
        curve_set = _normalize_curve_set(band.get("curveSet"), band.get("luminanceCurve"))
        dialog = CurvesDialog(band.get("label", band_id), curve_set, self)
        self.curves_dialog = dialog

        def apply_live_curve(curves):
            band["luminanceMode"] = "curve"
            band["curveSet"] = _normalize_curve_set(curves)
            band["luminanceCurve"] = {"points": band["curveSet"]["L"]["points"]}
            self._set_luminance_curve_control_state(band)
            self.preview_timer.start(250)

        def finish_curve_edit(result):
            if result != QDialog.DialogCode.Accepted:
                band["luminanceMode"] = original_mode
                band["curveSet"] = original_curve_set
                band["luminanceCurve"] = {"points": original_curve_set["L"]["points"]}
            elif dialog.use_slider_requested:
                band["luminanceMode"] = "slider"
            else:
                band["luminanceMode"] = "curve"
                band["curveSet"] = dialog.curve_set()
                band["luminanceCurve"] = {"points": band["curveSet"]["L"]["points"]}

            self.curves_dialog = None
            self._set_curve_edit_lock(False)
            self.syncing = True
            self._sync_active_pass_controls()
            self.syncing = False
            self._schedule_preview_update()
            dialog.deleteLater()

        dialog.curveChanged.connect(apply_live_curve)
        dialog.finished.connect(finish_curve_edit)
        self._set_curve_edit_lock(True)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _set_curve_edit_lock(self, locked):
        for widget in (
            self.sensitivity_combo,
            self.pass_combo,
            self.pass_enabled_check,
            self.pass_name_edit,
            self.new_pass_btn,
            self.dup_pass_btn,
            self.del_pass_btn,
            self.preset_combo,
            self.apply_preset_btn,
            self.load_btn,
            self.save_btn,
            self.reset_btn,
        ):
            widget.setEnabled(not locked)
        self.apply_btn.setEnabled(not locked and self.apply_thread is None)
        if not locked:
            self.del_pass_btn.setEnabled(len(self.recipe["passes"]) > 1)
            self.new_pass_btn.setEnabled(len(self.recipe["passes"]) < 4)

    def _sync_from_recipe(self):
        self.syncing = True
        r = normalize_recipe(self.recipe)
        self.recipe = r
        self._set_sensitivity_ranges(r["sensitivity"])
        self.image_type_combo.setCurrentIndex(1 if r["imageType"] == "starless" else 0)
        self.sensitivity_combo.setCurrentText(r["sensitivity"])
        self.global_strength_spin.setValue(r["globalStrength"])
        pc = r["protectionControls"]
        self.protect_stars_check.setChecked(pc["protectStars"])
        self.low_sat_protection_spin.setValue(pc["protectLowSaturation"])
        self.star_strength_combo.setCurrentText(pc["starMaskStrength"])
        self.pass_combo.blockSignals(True)
        self.pass_combo.clear()
        for p in r["passes"]:
            self.pass_combo.addItem(p.get("label", p.get("name", p["id"])), p["id"])
        self.pass_index = min(self.pass_index, len(r["passes"]) - 1)
        self.pass_combo.setCurrentIndex(self.pass_index)
        self.pass_combo.blockSignals(False)
        self._sync_active_pass_controls()
        self._update_band_soften_enabled()
        self.syncing = False
        self._schedule_preview_update()

    def _set_sensitivity_ranges(self, sensitivity):
        ranges = SENSITIVITY_RANGES.get(sensitivity, SENSITIVITY_RANGES["Normal"])
        for key in ("hueShift", "saturation", "luminance"):
            limit = ranges[key]
            for spin in self.band_controls[key].values():
                value = float(np.clip(spin.value(), -limit, limit))
                spin.setRange(-limit, limit)
                spin.setValue(value)
        self.neutral_spin.setRange(-ranges["neutral"], ranges["neutral"])

    def _update_band_soften_enabled(self):
        for spin in self.band_controls["soften"].values():
            for widget in getattr(spin, "_acm_row_widgets", (spin,)):
                widget.setVisible(True)

    def _sync_active_pass_controls(self):
        p = self._active_pass()
        self.pass_enabled_check.setChecked(p.get("enabled", True))
        self.pass_name_edit.setText(p.get("label", p.get("name", "")))
        self.selected_band_combo.setCurrentIndex(max(0, next((i for i, bd in enumerate(BAND_DEFS) if bd["id"] == p.get("selectedBandId", "red")), 0)))
        for b in p["bands"]:
            self.band_controls["hueShift"][b["id"]].setValue(b["hueShift"])
            self.band_controls["saturation"][b["id"]].setValue(b["saturation"])
            self.band_controls["luminance"][b["id"]].setValue(b["luminance"])
            self._set_luminance_curve_control_state(b)
            if b.get("custom") or b.get("id") == "custom":
                cm = b.get("customMask", {})
                self.band_controls["customStart"][b["id"]].setValue(cm.get("startHue", 300.0))
                self.band_controls["customEnd"][b["id"]].setValue(cm.get("endHue", 60.0))
                self.band_controls["customStrength"][b["id"]].setValue(cm.get("strength", 0.75))
                self.band_controls["customFeather"][b["id"]].setValue(cm.get("feather", 0.0))
                self.band_controls["customType"][b["id"]].setCurrentText(cm.get("type", "Chrominance"))
                self.band_controls["customWheel"][b["id"]].set_range(cm.get("startHue", 300.0), cm.get("endHue", 60.0))
            else:
                self.band_controls["width"][b["id"]].setValue(b["width"])
                self.band_controls["feather"][b["id"]].setValue(b["feather"])
            self.band_controls["soften"][b["id"]].setValue(b["maskSoftenRadius"])
        self.neutral_spin.setValue(p["neutralLuminance"]["luminance"])
        rm = p["rangeMask"]
        self.range_enabled_check.setChecked(rm.get("enabled", False))
        self.range_preset_combo.setCurrentText(rm.get("preset", "All") if rm.get("preset", "All") in list(RANGE_PRESETS.keys()) else "Custom")
        self.range_low_spin.setValue(rm.get("low", 0.0))
        self.range_high_spin.setValue(rm.get("high", 1.0))
        self.range_feather_spin.setValue(rm.get("feather", 0.10))
        self.range_soften_spin.setValue(rm.get("maskSoftenRadius", 0))
        self.range_boost_check.setChecked(rm.get("boostEnabled", False))
        self.del_pass_btn.setEnabled(len(self.recipe["passes"]) > 1)
        self.new_pass_btn.setEnabled(len(self.recipe["passes"]) < 4)

    def _write_globals_to_recipe(self):
        if getattr(self, "syncing", False):
            return
        self.recipe["imageType"] = "starless" if self.image_type_combo.currentIndex() == 1 else "stars"
        self.recipe["sensitivity"] = self.sensitivity_combo.currentText()
        self.recipe["globalStrength"] = self.global_strength_spin.value()
        self.recipe["protectionControls"] = {
            "protectStars": self.protect_stars_check.isChecked(),
            "protectLowSaturation": self.low_sat_protection_spin.value(),
            "starMaskStrength": self.star_strength_combo.currentText(),
        }

    def _write_active_pass_from_controls(self):
        if getattr(self, "syncing", False):
            return
        p = self._active_pass()
        p["enabled"] = self.pass_enabled_check.isChecked()
        p["label"] = self.pass_name_edit.text().strip() or p["id"]
        p["name"] = p["label"]
        p["selectedBandId"] = self.selected_band_combo.currentData()
        for b in p["bands"]:
            bid = b["id"]
            b["hueShift"] = self.band_controls["hueShift"][bid].value()
            b["saturation"] = self.band_controls["saturation"][bid].value()
            b["luminance"] = self.band_controls["luminance"][bid].value()
            if b.get("custom") or bid == "custom":
                b["customMask"] = {
                    "startHue": self.band_controls["customStart"][bid].value(),
                    "endHue": self.band_controls["customEnd"][bid].value(),
                    "strength": self.band_controls["customStrength"][bid].value(),
                    "feather": self.band_controls["customFeather"][bid].value(),
                    "type": self.band_controls["customType"][bid].currentText(),
                }
            else:
                b["width"] = self.band_controls["width"][bid].value()
                b["feather"] = self.band_controls["feather"][bid].value()
            b["maskSoftenRadius"] = self.band_controls["soften"][bid].value()
        p["neutralLuminance"]["luminance"] = self.neutral_spin.value()
        p["rangeMask"] = {
            "enabled": self.range_enabled_check.isChecked(),
            "low": self.range_low_spin.value(),
            "high": self.range_high_spin.value(),
            "feather": self.range_feather_spin.value(),
            "preset": self.range_preset_combo.currentText(),
            "maskSoftenRadius": self.range_soften_spin.value(),
            "boostEnabled": self.range_boost_check.isChecked(),
        }

    def _on_global_changed(self, *_args):
        self._write_globals_to_recipe()
        self._update_band_soften_enabled()
        self._schedule_preview_update()

    def _on_sensitivity_changed(self, *_args):
        self._write_globals_to_recipe()
        self.recipe = normalize_recipe(self.recipe)
        self._sync_from_recipe()

    def _on_pass_selected(self, idx):
        if idx < 0 or getattr(self, "syncing", False):
            return
        self._write_active_pass_from_controls()
        self.pass_index = idx
        self.syncing = True
        self._sync_active_pass_controls()
        self._update_band_soften_enabled()
        self.syncing = False
        self._schedule_preview_update()

    def _on_pass_enabled_changed(self, *_args):
        self._write_active_pass_from_controls()
        self._schedule_preview_update()

    def _on_pass_name_changed(self):
        self._write_active_pass_from_controls()
        self._sync_from_recipe()

    def _on_selected_band_changed(self, *_args):
        self._write_active_pass_from_controls()
        self._schedule_preview_update()

    def _on_band_control_changed(self, *_args):
        self._write_active_pass_from_controls()
        self._schedule_preview_update()

    def _on_neutral_changed(self, *_args):
        self._write_active_pass_from_controls()
        self._schedule_preview_update()

    def _on_range_changed(self, *_args):
        self._write_active_pass_from_controls()
        self._schedule_preview_update()

    def _on_range_preset_changed(self, name):
        if getattr(self, "syncing", False):
            return
        if name in RANGE_PRESETS:
            low, high = RANGE_PRESETS[name]
            self.range_low_spin.setValue(low)
            self.range_high_spin.setValue(high)
        self._write_active_pass_from_controls()
        self._schedule_preview_update()

    def _new_pass(self):
        if len(self.recipe["passes"]) >= 4:
            return
        n = len(self.recipe["passes"]) + 1
        self.recipe["passes"].append(default_pass(f"pass-{int(time.time())}", f"Pass {n}"))
        self.pass_index = len(self.recipe["passes"]) - 1
        self._sync_from_recipe()

    def _duplicate_pass(self):
        if len(self.recipe["passes"]) >= 4:
            return
        src = deepcopy(self._active_pass())
        src["id"] = f"pass-{int(time.time())}"
        src["label"] = src.get("label", "Pass") + " Copy"
        src["name"] = src["label"]
        self.recipe["passes"].append(src)
        self.pass_index = len(self.recipe["passes"]) - 1
        self._sync_from_recipe()

    def _delete_pass(self):
        if len(self.recipe["passes"]) <= 1:
            return
        del self.recipe["passes"][self.pass_index]
        self.pass_index = max(0, self.pass_index - 1)
        self._sync_from_recipe()

    def _reset_recipe(self):
        self.recipe = default_recipe()
        self.pass_index = 0
        self._sync_from_recipe()

    def _apply_builtin_preset(self):
        name = self.preset_combo.currentText()
        try:
            self.recipe = preset_recipe(name)
            self.pass_index = 0
            self._sync_from_recipe()
            self.status_label.setText(f"Loaded preset: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Preset failed", str(e))

    def _load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Astro Color Mixer JSON", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.recipe = normalize_recipe(json.load(f))
            self.pass_index = 0
            self._sync_from_recipe()
            self.status_label.setText(f"Loaded {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def _save_recipe(self):
        self._write_globals_to_recipe()
        self._write_active_pass_from_controls()
        path, _ = QFileDialog.getSaveFileName(self, "Save Astro Color Mixer JSON", "AstroColorMixer_Recipe.json", "JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(normalize_recipe(self.recipe), f, indent=2)
            self.status_label.setText(f"Saved {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _refresh_preview_source(self):
        try:
            with self.siril.image_lock():
                img = self.siril.get_image()
                raw = np.asarray(img.data, dtype=np.float32)
                display_bounds = _normalize_range(
                    np.asarray([img.mini, img.maxi], dtype=np.float32)
                )
                if display_bounds[1] > display_bounds[0]:
                    self.preview_display_range = (
                        float(display_bounds[0]), float(display_bounds[1])
                    )
                else:
                    self.preview_display_range = None
            data, _layout = _to_hwc(raw)
            data = _ensure_rgb(_normalize_range(data))
            data = np.flipud(data)
            self.proxy_data, self.preview_spatial_scale = _downsample_area(data, self.PREVIEW_MAX_DIM)
            self.proxy_data = self.proxy_data.astype(np.float32, copy=False)
            self.original_pixmap_base = _array_to_qpixmap(
                self.proxy_data, self.preview_display_range
            )
            self.status_label.setText(f"Preview proxy {self.proxy_data.shape[1]}x{self.proxy_data.shape[0]}")
            self._update_preview()
        except Exception as e:
            self.proxy_data = None
            self.preview_spatial_scale = 1.0
            self.original_pixmap_base = None
            self.processed_pixmap_base = None
            self.preview_label.setText("No preview available")
            self.status_label.setText(str(e))

    def _schedule_preview_update(self, *_args):
        self.preview_timer.start(200)

    def _update_preview(self):
        if self.proxy_data is None:
            return
        self._write_globals_to_recipe()
        self._write_active_pass_from_controls()
        try:
            mode = self.preview_mode_combo.currentText()
            preview_recipe = _scaled_preview_recipe(self.recipe, self.preview_spatial_scale)
            mask_modes = {
                "Band Mask": ("bandMask", (1.0, 0.40, 0.25)),
                "Range Mask": ("rangeMask", (0.25, 0.65, 1.0)),
                "Combined Mask": ("combinedMask", (1.0, 0.35, 0.95)),
                "Star Protection Mask": ("starMask", (0.75, 0.95, 1.0)),
            }
            if mode == "Original":
                result = self.proxy_data
                display_range = self.preview_display_range
                self.mask_stats_label.setText("")
            elif mode in mask_modes:
                mask_mode, tint = mask_modes[mode]
                mask_rgb = compute_preview_mask(self.proxy_data, preview_recipe, self.pass_index, self.selected_band_combo.currentData(), mask_mode)
                mask_values = _mask_values_from_rgb(mask_rgb)
                self.mask_stats_label.setText(_mask_stats_text(mask_values))
                if self.mask_overlay_check.isChecked():
                    opacity = self.mask_opacity_slider.value() / 100.0
                    result = _mask_overlay_image(self.proxy_data, mask_values, tint, opacity)
                    display_range = self.preview_display_range
                else:
                    result = mask_rgb
                    display_range = None
            else:
                result = process_image(self.proxy_data, preview_recipe)
                display_range = self.preview_display_range
                self.mask_stats_label.setText("")
            self.processed_pixmap_base = _array_to_qpixmap(result, display_range)
            self.status_label.setText("Preview updated")
            self._render_preview()
        except Exception as e:
            self.processed_pixmap_base = None
            self.preview_label.setText(f"Preview error:\n{e}")
            self.status_label.setText(str(e))

    def _render_preview(self):
        base = self.original_pixmap_base if self.show_original_active else self.processed_pixmap_base
        if base is None or base.isNull():
            return
        target_w = max(1, int(base.width() * self.preview_zoom))
        target_h = max(1, int(base.height() * self.preview_zoom))
        scaled = base.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.preview_label.resize(scaled.size())

    def _on_wheel_zoom(self, delta):
        self.preview_zoom = min(8.0, max(0.1, self.preview_zoom * (1.1 if delta > 0 else 1.0 / 1.1)))
        self._update_zoom_label()
        self._render_preview()

    def _zoom_in(self):
        self.preview_zoom = min(8.0, self.preview_zoom * 1.25)
        self._update_zoom_label()
        self._render_preview()

    def _zoom_out(self):
        self.preview_zoom = max(0.1, self.preview_zoom / 1.25)
        self._update_zoom_label()
        self._render_preview()

    def _zoom_reset(self):
        self.preview_zoom = 1.0
        self._update_zoom_label()
        self._render_preview()

    def _zoom_fit(self):
        base = self.original_pixmap_base if self.show_original_active else self.processed_pixmap_base
        if base is None or base.isNull():
            return
        viewport = self.preview_scroll.viewport().size()
        available_w = max(1, viewport.width() - 2)
        available_h = max(1, viewport.height() - 2)
        self.preview_zoom = min(
            available_w / max(1, base.width()),
            available_h / max(1, base.height()),
            8.0,
        )
        self.preview_zoom = max(0.01, self.preview_zoom)
        self._update_zoom_label()
        self._render_preview()

    def _update_zoom_label(self):
        self.zoom_label.setText(f"{round(self.preview_zoom * 100)}%")

    def _show_original_pressed(self):
        self.show_original_active = True
        self._render_preview()

    def _show_original_released(self):
        self.show_original_active = False
        self._render_preview()

    def closeEvent(self, event):
        if self.apply_thread is not None and self.apply_thread.isRunning():
            event.ignore()
            self.close_when_finished = True
            self.status_label.setText("Apply is still running. This window will close when it finishes.")
            QMessageBox.information(
                self,
                "Cannot close while Processing",
                "Cannot close while Processing",
            )
            return
        if self.curves_dialog is not None:
            self.curves_dialog.reject()
        super().closeEvent(event)

    def _apply_to_siril(self):
        if self.apply_thread is not None:
            return
        self._write_globals_to_recipe()
        self._write_active_pass_from_controls()
        recipe = normalize_recipe(self.recipe)
        try:
            with self.siril.image_lock():
                img = self.siril.get_image()
                raw = np.array(img.data, dtype=np.float32, copy=True)
            self.siril.log(f"Astro Color Mixer: copied raw pixel array shape={raw.shape}; processing in background.")
        except Exception as e:
            QMessageBox.critical(self, "Astro Color Mixer Error", str(e))
            try:
                self.siril.log(f"Astro Color Mixer error: {e}")
            except Exception:
                pass
            return

        self.apply_btn.setEnabled(False)
        self.status_label.setText("Applying in background...")
        try:
            self.siril.log("Astro Color Mixer timing progress:")
        except Exception:
            pass
        self.apply_started_at = time.time()
        self.apply_thread = QThread(self)
        self.apply_worker = AstroColorMixerApplyWorker(raw, recipe)
        self.apply_worker.moveToThread(self.apply_thread)
        self.apply_thread.started.connect(self.apply_worker.run)
        self.apply_worker.finished.connect(self._on_apply_finished)
        self.apply_worker.failed.connect(self._on_apply_failed)
        self.apply_worker.progress.connect(self._on_apply_progress)
        self.apply_worker.finished.connect(self.apply_thread.quit)
        self.apply_worker.failed.connect(self.apply_thread.quit)
        self.apply_worker.finished.connect(self.apply_worker.deleteLater)
        self.apply_worker.failed.connect(self.apply_worker.deleteLater)
        self.apply_thread.finished.connect(self._clear_apply_worker)
        self.apply_thread.start()


    def _on_apply_progress(self, message):
        try:
            self.siril.log(f"  {message}")
        except Exception:
            pass
        self.status_label.setText(message)

    def _on_apply_finished(self, payload):
        timings = []
        pixeldata = payload
        if isinstance(payload, dict):
            pixeldata = payload.get("pixeldata")
            timings = payload.get("timings", []) or []
        try:
            write_start = time.perf_counter()
            with self.siril.image_lock():
                self.siril.undo_save_state("Astro Color Mixer")
                self.siril.set_image_pixeldata(pixeldata)
            write_elapsed = time.perf_counter() - write_start
            elapsed = time.time() - getattr(self, "apply_started_at", time.time())
            self.siril.log(f"Astro Color Mixer applied in {elapsed:.2f}s.")
            self.siril.log(f"  write result to Siril: {write_elapsed:.3f}s")
            self.status_label.setText("Applied to Siril")
            self._refresh_preview_source()
        except Exception as e:
            QMessageBox.critical(self, "Astro Color Mixer Error", str(e))
            try:
                self.siril.log(f"Astro Color Mixer write error: {e}")
            except Exception:
                pass
    def _on_apply_failed(self, message):
        QMessageBox.critical(self, "Astro Color Mixer Error", message)
        try:
            self.siril.log(f"Astro Color Mixer processing error: {message}")
        except Exception:
            pass

    def _clear_apply_worker(self):
        if self.apply_thread is not None:
            self.apply_thread.deleteLater()
        self.apply_thread = None
        self.apply_worker = None
        self.apply_btn.setEnabled(self.curves_dialog is None)
        if self.close_when_finished:
            self.close_when_finished = False
            QTimer.singleShot(0, self.close)


def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except SirilConnectionError as e:
        print(f"Could not connect to Siril: {e}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = AstroColorMixerWindow(siril)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
