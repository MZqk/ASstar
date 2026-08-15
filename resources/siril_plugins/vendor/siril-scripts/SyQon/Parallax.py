# (c) Adrian Knagg-Baugh 2026
# (c) SyQon 2026
# (c) Franklin Marek 2026 (Statistical Stretch)
# SPDX-License-Identifier: MIT
"""
    ███████╗██╗   ██╗ ██████╗  ██████╗ ███╗   ██╗
    ██╔════╝╚██╗ ██╔╝██╔═══██╗██╔═══██╗████╗  ██║
    ███████╗ ╚████╔╝ ██║   ██║██║   ██║██╔██╗ ██║
    ╚════██║  ╚██╔╝  ██║▄▄ ██║██║   ██║██║╚██╗██║
    ███████║   ██║   ╚██████╔╝╚██████╔╝██║ ╚████║
    ╚══════╝   ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═══╝
    ★  P A R A L L A X   —   O F F I C I A L   E D I T I O N  ★
    ┌──────────────────────────────────────────────────┐
    │            Siril Edition v1.2.7                  │
    │  Aberration Correction · Star Reduction ·        │
    │              Sharpening                          │
    │         https://syqon.eu/parallax                │
    └──────────────────────────────────────────────────┘

Official SyQon Parallax script for Siril — self-contained single file edition.

Three embedded models:
  • parallax_correction.pth  — StellarDirectNet aberration correction
  • parallax_star_reduction.pth — StellarDirectNet level-conditioned star reduction
  • parallax_sharpen.pth     — AstroNAFLiteDeblur sharpening

Model weights are placed in <siril_userdata>/syqon_parallax/.
Install them manually via the GUI (Install Model buttons).

Usage:
    GUI mode (default):
        Add this script to the Scripts menu and run it.
"""

# Version History:
# =================
# v1.0.0 Initial release (self-contained single-file Parallax Official Edition).
#       Three embedded models: aberration correction (StellarDirectNet),
#       level-conditioned star reduction (StellarDirectNet) and sharpening
#       (AstroNAFLiteDeblur), with GUI and programmatic (CLI) modes.
# v1.0.1 Added Nano / Pro dual-edition support: in-GUI Nano/Pro toggle,
#       per-edition model filenames and download URLs, "edition" config key
#       and --edition CLI argument, with edition-aware window/hero titles.
#       Added dynamic multi-threaded tiled inference (_configure_cpu_performance
#       + ThreadPoolExecutor) that scales workers/threads to the host CPU.
#       Added hybrid neural-morphological star reduction for Pro levels 7-10
#       (circular erosion + soft star-mask blending) and per-edition star
#       level range (max 6 Nano / 7 Pro) with value clamping.
#       Removed preview downscaling in _numpy_to_qimage (full-resolution preview).
#       Renamed the "Cancel" button to "Stop" and reordered the action buttons.
#       Updated the sharpening warning text for values above 2.0.
# v1.0.2 Updated Nano model filenames (f_nano_*.pt/.pth).
#       Reworked hybrid neural-morphological star reduction into a unified,
#       parameterised pipeline applied across all star levels (1-7) via a
#       star_settings table (erosion size, threshold, slope, blur kernel,
#       passes) with fractional-level blend factor and a 0.30 blend scaling
#       for non-Pro editions.
# v1.2.0 Updated script and GUI to v1.2.
#       Enhanced the user interface (UI) design.
#       Added the "Defined" pipeline mode with 3 new neural models, and updated the existing models in "Classic (Natural)" mode.
#       Updated all purchase and download URLs to point to syqon.eu instead of syqon.it.
# v1.2.1 Performance optimization and star core preservation bug fixes.
#       Optimized median calculation speed by 60x with spatial striding.
#       Ported tiling blend/accumulator logic to native PyTorch CPU/GPU tensors to avoid numpy/GIL overhead.
#       Added dynamic input-output chromatic filter to prevent magenta/purple cores on saturated or mid-tone stars.
#       Moved "Enable real-time preview updates" from Advanced to Tuning and added warning.
#       Cleaned up GPU pre-flight tests.
# v1.2.2 Fixed Aesthetics checkpoint reconstruction and strict weight loading.
#       Restored the training-compatible LayerNorm layout, added dark-sky RGB
#       drift protection, FP16 inference on native GPUs, streaming tile
#       preparation, device-aware caches, and throttled full-frame previews.
# v1.2.3 Restored the full enabled pipeline order after star-reduction-only tests:
#       Aberration Correction -> Sharpen Star -> Deblur Nebula.
# v1.2.4 Added automatic large-image tuning: memory-aware larger GPU tiles,
#       tile-area-aware batching, and disabled costly full-frame live previews
#       while processing images above 24 megapixels.
# v1.2.5 Optimized ROI previews to process and return only the selected crop.
#       Expanded native GPU selection for CUDA/ROCm, Intel XPU, Apple MPS and
#       Windows DirectML (AMD/Intel), with backend-aware precision and caches.
# v1.2.6 Bug Fix
# v1.2.7 Consolidated production update.
#       Fixed Defined/Aesthetics inference reconstruction by reading checkpoint
#       arguments correctly, restoring the training-compatible LayerNorm
#       layout and enforcing strict model-weight loading.
#       Preserved dark-sky RGB neutrality to prevent green/blue background
#       shifts and enabled backend-aware FP16 inference, streaming tile setup
#       and device-aware model caches.
#       Improved large-image performance with automatic GPU tile geometry and
#       tile-area-aware batching; live full-frame updates are disabled above
#       24 MP with a clear English performance notice. Users can now explicitly
#       keep live preview enabled for a large image at their own performance risk.
#       Reworked Preview ROI to process only the selected crop, use a
#       single-tile geometry where possible, avoid full-frame allocations and
#       keep it responsive on high-resolution images.
#       Expanded acceleration support and fallback handling for NVIDIA CUDA,
#       AMD ROCm/HIP, Apple Metal MPS, Intel XPU and Windows DirectML.
#       Limited the Advanced safety lock to tile size, overlap and edge pad;
#       batch, CPU/GPU and stretch controls stay independently usable.
#       Separated Pro Classic and Defined model installation into six explicit
#       slots (three per pipeline), restored the visible Classic Sharpen Star
#       slot, and strictly validate every selected checkpoint architecture
#       before it can overwrite a destination. This prevents Classic/Defined
#       filename confusion and model/shape-mismatch loading errors.
#       Restored the original three-model Classic chain (Correction, Sharpen
#       Star and Nebula Deblur) as an isolated pipeline, independent from the
#       Defined checkpoints and their NAFNet architecture.
#       Reworked Classic star reduction at all levels to use its neural mask
#       with bounded circular luminance reduction. High levels now reduce
#       progressively while retaining round, colour-preserving stellar cores
#       instead of repeated RGB erosions that could turn level-7 stars black.
#       Added a Classic-only Nebula Deblur protection mask: it attenuates the
#       deblur residual on compact stellar peaks and isolated artefacts while
#       preserving coherent extended nebulosity, with stronger protection at
#       higher deblur strengths.
#       Set Defined as the default Pro pipeline for new and legacy
#       configurations that do not yet contain a saved pipeline mode.
#       Removed the optional FITS export/reload and output-folder controls.
#       Apply to Siril now remains an in-memory handoff with no file-format
#       conversion; the original data type, precision and FITS orientation are
#       restored when the result is applied to the loaded Siril image.
#       Updated processing order to: Aberration Correction -> Sharpen Star ->
#       Nebula Deblur, so non-stellar deblur is applied last.
#       Refined the GUI into a compact graphite desktop theme, removed
#       unintended opaque layout rectangles, and improved tabs, controls,
#       model status rows, preview toolbar and action/status presentation.
#       Removed unsupported window-opacity animation and forced an opaque GUI
#       backing surface; live updates now use queued repaint scheduling to
#       prevent transparency artefacts and flicker in embedded Linux/Siril Qt.
#       Made the GUI resolution-aware: startup size now follows available
#       screen space, the controls/preview use a resizable splitter, and labels
#       and progress/status areas can shrink cleanly on compact displays.

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Tuple

# ============================================================================
# sirilpy bootstrap
# ============================================================================

import sirilpy as s

s.ensure_installed("PySide6", "astropy")

if platform.system() == "Darwin" and platform.machine() != "arm64":
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0",
                               "--quiet", "--exists-action", "i"])
    except subprocess.CalledProcessError:
        pass

th = s.TorchHelper()
th.ensure_torch()

if sys.platform == "win32":
    try:
        import torch as _t
        if not _t.cuda.is_available() and not (hasattr(_t, "xpu") and _t.xpu.is_available()):
            s.ensure_installed("torch-directml")
    except Exception:
        pass

# ============================================================================
# Third-party imports
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PySide6.QtCore import QObject, QThread, Qt, Signal, QTimer, QEvent, QVariantAnimation
from PySide6.QtGui import (QDesktopServices, QImage, QPainter, QPen, QColor,
                           QPixmap, QFont, QCursor, QBrush, QPainterPath)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
    QSlider, QSizePolicy, QFrame, QGroupBox, QTabWidget, QTabBar,
    QScrollArea, QSplitter,
)

# ============================================================================
# Constants
# ============================================================================

SCRIPT_VERSION   = "1.2.7"
PARALLAX_BUY_URL = "https://syqon.eu/parallax"

_ENGINE_DIR: Optional[Path] = None
_CURRENT_EDITION = "nano"


def _resolve_engine_dir(siril) -> Path:
    global _ENGINE_DIR
    if _ENGINE_DIR is not None:
        return _ENGINE_DIR
    try:
        user_dir = siril.get_siril_userdatadir() if siril else str(Path.home() / ".siril")
    except Exception:
        user_dir = str(Path.home() / ".siril")
    _ENGINE_DIR = Path(user_dir) / "syqon_parallax"
    _ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    return _ENGINE_DIR


def _correction_path(mode="classic") -> Path:
    if _CURRENT_EDITION == "nano":
        return _ENGINE_DIR / "f_nano_corrector.pt"
    if mode == "aesthetics":
        return _ENGINE_DIR / "aesthetics_staronly.pth"
    return _ENGINE_DIR / "parallax_correction.pth"


def _star_reduce_path(mode="classic") -> Path:
    if _CURRENT_EDITION == "nano":
        return _ENGINE_DIR / "f_nano_reduce.pth"
    if mode == "aesthetics":
        return _ENGINE_DIR / "aesthetics_starreduction.pth"
    return _ENGINE_DIR / "parallax_star_reduction.pth"


def _sharpen_path(mode="classic") -> Path:
    if _CURRENT_EDITION == "nano":
        return _ENGINE_DIR / "f_nano_sharp.pth"
    if mode == "aesthetics":
        return _ENGINE_DIR / "aesthetics_deblur.pth"
    return _ENGINE_DIR / "parallax_sharpen.pth"

# ============================================================================
# JSON Config
# ============================================================================

DEFAULT_CONFIG = {
    "edition":       "nano",
    # Pro starts on Defined; Classic remains selectable from Pipeline Mode.
    "mode":          "aesthetics",
    "correct":       True,
    "star_level":    3,        # 0 = off, 1-6
    "sharpen":       True,
    "sharpen_alpha": 1.0,
    "tile_size":     512,
    "overlap":       64,
    "pad":           96,
    "use_mtf":       True,
    "mtf_target":    0.25,
    "linked_stretch": False,
    "batch_size":    "Auto",
    "allow_large_live_preview": False,
}

import subprocess

def get_mac_memory_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
        bytes_val = int(out.strip())
        return bytes_val / (1024.0 ** 3)
    except Exception:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024.0 ** 3)
        except Exception:
            return 8.0

def get_auto_batch_size() -> int:
    gb = get_mac_memory_gb()
    if gb >= 30.0:
        return 5  # 32 GB or more -> batch 5
    elif gb >= 20.0:
        return 4  # 24 GB -> batch 4
    elif gb >= 15.0:
        return 3  # 16 GB -> batch 3
    elif gb >= 7.0:
        return 2  # 8 GB -> batch 2
    return 1      # fallback


_LARGE_IMAGE_PIXELS = 24_000_000


def _resolve_inference_batch_size(total: int, tile: int, device: torch.device,
                                  batch_size_cfg: str) -> int:
    if device.type == "cpu":
        return 1
    cfg_val = str(batch_size_cfg).strip()
    if cfg_val == "Auto":
        base = get_auto_batch_size()
        # The old heuristic assumed 512x512 tiles. Keep approximately the same
        # activation-memory budget when large-image mode raises the tile size.
        area_scale = (512.0 / max(float(tile), 1.0)) ** 2
        batch = max(1, int(base * area_scale))
    else:
        try:
            batch = max(1, int(cfg_val))
        except ValueError:
            batch = 1
    return min(max(1, total), batch)


def _tune_large_image_geometry(img: np.ndarray, tile: int, overlap: int,
                               device: torch.device) -> Tuple[int, int, bool]:
    """Use fewer, larger tiles on sufficiently capable native GPUs."""
    h, w = img.shape[:2]
    pixels = h * w
    is_large = pixels >= _LARGE_IMAGE_PIXELS
    if not is_large or device.type not in ("cuda", "mps"):
        return tile, overlap, is_large

    target_tile = tile
    if device.type == "mps":
        memory_gb = get_mac_memory_gb()
        if memory_gb >= 30.0 and pixels >= 50_000_000:
            target_tile = max(target_tile, 1024)
        elif memory_gb >= 15.0:
            target_tile = max(target_tile, 768)
    else:  # CUDA
        try:
            memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3)
        except Exception:
            memory_gb = 0.0
        if memory_gb >= 12.0 and pixels >= 50_000_000:
            target_tile = max(target_tile, 1024)
        elif memory_gb >= 8.0:
            target_tile = max(target_tile, 768)

    target_tile = max(8, (int(target_tile) // 8) * 8)
    return target_tile, min(overlap, target_tile - 8), is_large


def _config_path(siril) -> Path:
    try:
        cfg_dir = siril.get_siril_configdir() if siril else str(Path.home() / ".siril")
    except Exception:
        cfg_dir = str(Path.home() / ".siril")
    return Path(cfg_dir) / "syqon_parallax_config.json"


def load_config(siril) -> dict:
    p = _config_path(siril)
    if not p.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with open(p) as f:
            loaded = json.load(f)
        return {**DEFAULT_CONFIG, **loaded}
    except Exception as exc:
        print(f"Warning: could not load config: {exc}")
        return dict(DEFAULT_CONFIG)


def save_config(cfg: dict, siril) -> None:
    p = _config_path(siril)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as exc:
        print(f"Warning: could not save config: {exc}")

# ============================================================================
# Image preparation helpers  (Siril planar ↔ HWC float32)
# ============================================================================

def _prepare_for_inference(data: np.ndarray, flip_vertical: bool = False):
    """Siril planar (C,H,W) → HWC float32 [0,1]. Returns (xrgb, orig_dtype, scale, orig_was_mono)."""
    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.integer):
        scale = float(np.iinfo(original_dtype).max)
        x = data.astype(np.float32) / scale
    else:
        x = data.astype(np.float32)
        scale = 1.0

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    is_planar = (x.ndim == 3 and x.shape[0] in (1, 3, 4)
                 and x.shape[0] < min(x.shape[1], x.shape[2]))
    if is_planar:
        x = np.transpose(x, (1, 2, 0))

    orig_was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    if x.ndim == 2:
        xrgb = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        xrgb = np.repeat(x, 3, axis=2)
    else:
        xrgb = x[..., :3].copy()

    if flip_vertical:
        xrgb = np.flip(xrgb, axis=0)

    return xrgb.astype(np.float32, copy=False), original_dtype, scale, orig_was_mono


def _slice_raw_roi(data: np.ndarray, roi: Tuple[int, int, int, int],
                   flip_vertical: bool = False) -> np.ndarray:
    """Slice a display-coordinate ROI before float/RGB conversion.

    This avoids converting and copying a giant source image merely to process a
    small preview crop. FITS bottom-up orientation is accounted for while the
    data is still in its original planar/HWC representation.
    """
    x, y, w, h = (int(v) for v in roi)
    is_planar = (data.ndim == 3 and data.shape[0] in (1, 3, 4)
                 and data.shape[0] < min(data.shape[1], data.shape[2]))
    if data.ndim == 2:
        image_h, image_w = data.shape
    elif is_planar:
        image_h, image_w = data.shape[1], data.shape[2]
    else:
        image_h, image_w = data.shape[0], data.shape[1]

    x = max(0, min(image_w - 1, x))
    y = max(0, min(image_h - 1, y))
    w = max(1, min(image_w - x, w))
    h = max(1, min(image_h - y, h))
    source_y = image_h - (y + h) if flip_vertical else y

    if data.ndim == 2:
        return data[source_y:source_y + h, x:x + w]
    if is_planar:
        return data[:, source_y:source_y + h, x:x + w]
    return data[source_y:source_y + h, x:x + w, ...]


def _restore_dtype(data: np.ndarray, original_dtype: np.dtype, scale: float) -> np.ndarray:
    if original_dtype == np.float32:
        return data
    if np.issubdtype(original_dtype, np.integer):
        return np.clip(data * scale, 0, scale).astype(original_dtype)
    return data.astype(original_dtype)

# ============================================================================
# MTF stretch / destretch (Statistical Stretch by Franklin Marek)
# ============================================================================

def _mtf_apply(x: np.ndarray, s: float, m: float, h: float) -> np.ndarray:
    denom = max(h - s, 1e-8)
    xp = np.clip((x - s) / denom, 0.0, 1.0).astype(np.float32, copy=False)
    num = (m - 1.0) * xp
    den = (2.0 * m - 1.0) * xp - m
    y = np.divide(num, den, out=np.zeros_like(xp), where=np.abs(den) > 1e-12)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def _mtf_inverse(y: np.ndarray, s: float, m: float, h: float) -> np.ndarray:
    yp = np.clip(y.astype(np.float32, copy=False), 0.0, 1.0)
    num = (((s + h) * m - s) * yp - s * m + s)
    den = (2.0 * m - 1.0) * yp - m + 1.0
    x = np.divide(num, den, out=np.full_like(yp, s), where=np.abs(den) > 1e-12)
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def _mtf_params_unlinked(img_rgb01: np.ndarray, targetbg: float = 0.15) -> dict:
    """Per-channel MTF params matching stretch_color_image(linked=False, no_black_clip=True)."""
    x = np.asarray(img_rgb01, dtype=np.float32)
    tb = float(np.clip(targetbg, 1e-6, 1.0 - 1e-6))
    s_arr = np.zeros(3, dtype=np.float32)
    m_arr = np.zeros(3, dtype=np.float32)
    for c in range(3):
        ch = x[..., c]
        bp = float(np.clip(ch.min(), 0.0, 1.0 - 1e-6))
        denom = max(1.0 - bp, 1e-12)
        med_rescaled = float(np.clip((float(np.median(ch)) - bp) / denom, 1e-6, 1.0 - 1e-6))
        denom_m = (2.0 * tb - 1.0) * med_rescaled - tb
        if abs(denom_m) < 1e-12:
            denom_m = 1e-12
        m = float(np.clip(med_rescaled * (tb - 1.0) / denom_m, 1e-4, 1.0 - 1e-4))
        s_arr[c] = bp
        m_arr[c] = m
    return {"s": s_arr, "m": m_arr, "h": np.ones(3, dtype=np.float32), "linked": False}


def _mtf_params_linked(img_rgb01: np.ndarray, targetbg: float = 0.15) -> dict:
    """Single blackpoint + avg-channel-median MTF matching stretch_color_image(linked=True, no_black_clip=True)."""
    x = np.asarray(img_rgb01, dtype=np.float32)
    tb = float(np.clip(targetbg, 1e-6, 1.0 - 1e-6))
    bp = float(np.clip(x.min(), 0.0, 1.0 - 1e-6))
    denom = max(1.0 - bp, 1e-12)
    # avg of per-channel medians (matches processColorImage Step 2 in StatisticalStretch)
    om0 = float(np.median(np.clip((x[..., 0] - bp) / denom, 0.0, 1.0)))
    om1 = float(np.median(np.clip((x[..., 1] - bp) / denom, 0.0, 1.0)))
    om2 = float(np.median(np.clip((x[..., 2] - bp) / denom, 0.0, 1.0)))
    med_rescaled = float(np.clip((om0 + om1 + om2) / 3.0, 1e-6, 1.0 - 1e-6))
    denom_m = (2.0 * tb - 1.0) * med_rescaled - tb
    if abs(denom_m) < 1e-12:
        denom_m = 1e-12
    m = float(np.clip(med_rescaled * (tb - 1.0) / denom_m, 1e-4, 1.0 - 1e-4))
    return {"s": bp, "m": m, "h": 1.0, "linked": True}


def apply_mtf_stretch(img_rgb01: np.ndarray, params: dict) -> np.ndarray:
    x = np.asarray(img_rgb01, dtype=np.float32)
    out = np.empty_like(x)
    if params["linked"]:
        s, m, h = float(params["s"]), float(params["m"]), float(params["h"])
        for c in range(3):
            out[..., c] = _mtf_apply(x[..., c], s, m, h)
    else:
        for c in range(3):
            out[..., c] = _mtf_apply(x[..., c],
                                      float(params["s"][c]),
                                      float(params["m"][c]),
                                      float(params["h"][c]))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_mtf_inverse(img_rgb01: np.ndarray, params: dict) -> np.ndarray:
    y = np.asarray(img_rgb01, dtype=np.float32)
    out = np.empty_like(y)
    if params["linked"]:
        s, m, h = float(params["s"]), float(params["m"]), float(params["h"])
        for c in range(3):
            out[..., c] = _mtf_inverse(y[..., c], s, m, h)
    else:
        for c in range(3):
            out[..., c] = _mtf_inverse(y[..., c],
                                        float(params["s"][c]),
                                        float(params["m"][c]),
                                        float(params["h"][c]))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

# ============================================================================
# ── BEGIN EMBEDDED MODEL ARCHITECTURES ──────────────────────────────────────
# (c) SyQon 2026
# ============================================================================

# ---------------------------------------------------------------------------
# StellarDirectNet  (aberration correction + star reduction)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(x).view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.se    = SEBlock(channels)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.se(self.conv2(out))
        return out + x


class StellarDirectNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_res_blocks=2,
                 coord_conv=True, cond_level=False, aggressive_correction=False):
        super().__init__()
        self.coord_conv   = coord_conv
        self.cond_level   = cond_level
        self.max_residual = 0.5 if aggressive_correction else 0.35
        extra    = (2 if coord_conv else 0) + (1 if cond_level else 0)
        input_ch = in_channels + extra

        self.enc1_conv  = nn.Conv2d(input_ch,          base_channels,     3, padding=1)
        self.enc1_res   = nn.Sequential(*[ResBlock(base_channels)     for _ in range(num_res_blocks)])
        self.down1      = nn.Conv2d(base_channels,      base_channels * 2, 4, stride=2, padding=1)
        self.enc2_res   = nn.Sequential(*[ResBlock(base_channels * 2) for _ in range(num_res_blocks)])
        self.down2      = nn.Conv2d(base_channels * 2,  base_channels * 4, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(*[ResBlock(base_channels * 4) for _ in range(num_res_blocks)])
        self.up2        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2_conv  = nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, 3, padding=1)
        self.dec2_res   = nn.Sequential(*[ResBlock(base_channels * 2) for _ in range(num_res_blocks)])
        self.up1        = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1_conv  = nn.Conv2d(base_channels * 2 + base_channels, base_channels, 3, padding=1)
        self.dec1_res   = nn.Sequential(*[ResBlock(base_channels)     for _ in range(num_res_blocks)])
        self.final_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, level=None):
        identity = x
        b, _, h, w = x.shape
        feats = [x]
        if self.coord_conv:
            yg = torch.linspace(-1., 1., h, device=x.device, dtype=x.dtype).view(1,1,h,1).expand(b,1,h,w)
            xg = torch.linspace(-1., 1., w, device=x.device, dtype=x.dtype).view(1,1,1,w).expand(b,1,h,w)
            feats += [yg, xg]
        if self.cond_level and level is not None:
            level_expanded = level.expand(b) if level.numel() == 1 else level
            feats.append(level_expanded.view(b,1,1,1).expand(b,1,h,w))
        inp = torch.cat(feats, dim=1)
        e1  = self.enc1_res(self.enc1_conv(inp))
        e2  = self.enc2_res(self.down1(e1))
        bn  = self.bottleneck(self.down2(e2))
        d2  = self.dec2_res(self.dec2_conv(torch.cat([self.up2(bn), e2], dim=1)))
        d1  = self.dec1_res(self.dec1_conv(torch.cat([self.up1(d2), e1], dim=1)))
        delta = torch.tanh(self.final_conv(d1)) * self.max_residual
        return torch.clamp(identity + delta, 0., 1.)


# ---------------------------------------------------------------------------
# AstroNAFLiteDeblur  (sharpening)
# ---------------------------------------------------------------------------

class LayerNorm2dClassic(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps
    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float().contiguous()
        mean = x_f32.mean(dim=1, keepdim=True)
        var  = (x_f32 - mean).pow(2).mean(dim=1, keepdim=True)
        out = (x_f32 - mean) / torch.sqrt(var + self.eps) * self.weight.float() + self.bias.float()
        return out.to(orig_dtype)

class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Keep the exact module/key layout used during training
        # (``norm.weight`` / ``norm.bias``), while making the permuted tensor
        # contiguous for DirectML and other backends that dislike strided LN.
        self.norm = nn.LayerNorm(num_features, eps=eps)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1); return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        return x * self.conv(self.pool(x))

class NAFBlockClassic(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout_rate=0.0):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand
        self.norm1    = LayerNorm2dClassic(channels)
        self.conv1    = nn.Conv2d(channels, dw_ch, 1, bias=True)
        self.conv2    = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.sg       = SimpleGate()
        self.sca      = SimplifiedChannelAttention(dw_ch // 2)
        self.conv3    = nn.Conv2d(dw_ch // 2, channels, 1, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.beta     = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2    = LayerNorm2dClassic(channels)
        self.conv4    = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.conv5    = nn.Conv2d(ffn_ch // 2, channels, 1, bias=True)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gamma    = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        res = x
        x = self.conv3(x * self.sca(self.sg(self.conv2(self.conv1(self.norm1(x))))))
        x = self.dropout1(x)
        y = res + x * self.beta
        x = self.sg(self.conv4(self.norm2(y)))
        x = self.dropout2(self.conv5(x))
        return y + x * self.gamma

class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout_rate=0.0):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand
        self.norm1    = LayerNorm2d(channels)
        self.conv1    = nn.Conv2d(channels, dw_ch, 1, bias=True)
        self.conv2    = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch, bias=True)
        self.sg       = SimpleGate()
        self.sca      = SimplifiedChannelAttention(dw_ch // 2)
        self.conv3    = nn.Conv2d(dw_ch // 2, channels, 1, bias=True)
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.beta     = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2    = LayerNorm2d(channels)
        self.conv4    = nn.Conv2d(channels, ffn_ch, 1, bias=True)
        self.conv5    = nn.Conv2d(ffn_ch // 2, channels, 1, bias=True)
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gamma    = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = identity + x * self.beta

        identity = y
        y = self.norm2(y)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return identity + y * self.gamma

class PixelShuffleUp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=True), nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

class BilinearUp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels // 2, 3, padding=1, bias=True)
    def forward(self, x):
        return self.proj(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))

class AstroNAFLiteDeblur(nn.Module):
    def __init__(self, img_channels=3, width=32, enc_blocks=[1, 2, 2],
                 middle_blocks=3, dec_blocks=[1, 1], dropout_rate=0.0,
                 upsample_mode="bilinear"):
        super().__init__()
        if len(dec_blocks) < len(enc_blocks):
            dec_blocks = list(dec_blocks) + [1] * (len(enc_blocks) - len(dec_blocks))
        elif len(dec_blocks) > len(enc_blocks):
            dec_blocks = dec_blocks[:len(enc_blocks)]
        self.intro         = nn.Conv2d(img_channels, width, 3, padding=1, bias=True)
        self.ending        = nn.Conv2d(width, img_channels, 3, padding=1, bias=True)
        self.upsample_mode = upsample_mode
        chan = width
        self.encoders, self.downs, self.ups, self.decoders = (nn.ModuleList() for _ in range(4))
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlockClassic(chan, dropout_rate=dropout_rate) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2, bias=True))
            chan *= 2
        self.middle = nn.Sequential(*[NAFBlockClassic(chan, dropout_rate=dropout_rate) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(PixelShuffleUp(chan) if upsample_mode == "pixelshuffle" else BilinearUp(chan))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlockClassic(chan, dropout_rate=dropout_rate) for _ in range(n)]))
        self.padder_size = 2 ** len(enc_blocks)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x

    def forward(self, inp):
        h, w = inp.shape[-2:]
        x    = self._pad(inp)
        res  = x
        x    = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = x + skip
            x = dec(x)
        x = self.ending(x) + res
        return x[..., :h, :w]


class NAFNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, width=32, middle_blk_num=4, enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downsamples.append(
                nn.Sequential(
                    nn.PixelUnshuffle(2),
                    nn.Conv2d(chan * 4, chan * 2, kernel_size=1, bias=False)
                )
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        _, _, H, W = inp.shape
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x = inp

        x_intro = self.intro(x)
        
        enc_feats = []
        x_enc = x_intro
        for encoder, downsample in zip(self.encoders, self.downsamples):
            x_enc = encoder(x_enc)
            enc_feats.append(x_enc)
            x_enc = downsample(x_enc)

        x_mid = self.middle_blks(x_enc)

        x_dec = x_mid
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(enc_feats)):
            x_dec = upsample(x_dec)
            x_dec = x_dec + skip
            x_dec = decoder(x_dec)

        x_out = self.ending(x_dec)
        x_out = x_out + x[:, :x_out.shape[1], :, :]

        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :, :H, :W]

        return x_out


def build_nafnet_lite(preset_name="balanced", upsample_mode="bilinear", img_channels=3):
    presets = {
        "lite":          {"width": 24, "enc_blocks": [1, 1, 2], "middle_blocks": 2, "dec_blocks": [1, 1]},
        "balanced":      {"width": 32, "enc_blocks": [1, 2, 2], "middle_blocks": 3, "dec_blocks": [1, 1]},
        "quality_light": {"width": 40, "enc_blocks": [2, 2, 3], "middle_blocks": 4, "dec_blocks": [2, 1]},
    }
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")
    cfg = presets[preset_name]
    return AstroNAFLiteDeblur(
        img_channels=img_channels, width=cfg["width"],
        enc_blocks=cfg["enc_blocks"], middle_blocks=cfg["middle_blocks"],
        dec_blocks=cfg["dec_blocks"], upsample_mode=upsample_mode,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def _load_pth(path: str) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, nn.Module):
        return {"model_state_dict": obj.state_dict()}
    raise RuntimeError(f"Unrecognised checkpoint format: {type(obj)}")


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ["model", "model_state_dict", "state_dict"]:
        if key in ckpt:
            val = ckpt[key]
            if isinstance(val, dict):
                return val
    return ckpt


def _clean_state_dict(sd: dict) -> dict:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(key.startswith("module.") for key in keys):
        sd = {key[7:]: value for key, value in sd.items()}
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def pick_device(use_gpu: bool = True) -> torch.device:
    if not use_gpu:
        return torch.device("cpu")
    # PyTorch exposes both NVIDIA CUDA and AMD ROCm/HIP through the ``cuda``
    # device API, so this branch supports either vendor's native runtime.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    # DirectML is the broad Windows fallback for AMD, Intel and older NVIDIA
    # adapters when a native CUDA/XPU runtime is unavailable.
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except (ImportError, AttributeError, RuntimeError):
        pass
    return torch.device("cpu")


def _empty_device_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "xpu" and hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()
    except Exception:
        pass


def _configure_cpu_performance(device: torch.device) -> int:
    """
    Dynamically configures PyTorch threading based on host CPU's capability and active edition.
    - If edition is "nano" (traditional): we run sequentially (1 worker) and set moderate threading.
    - If edition is "pro" and device is CPU:
      - We query total logical cores via `os.cpu_count()`.
      - If cores <= 4: "crappy CPU" -> use 2 threads (keeps system responsive, avoids lag).
      - If 4 < cores <= 8: "normal CPU" -> use 4 threads.
      - If cores > 8: "beast CPU" -> use 4 threads per worker and scale workers up to 7 to exploit the CPU.
    """
    if _CURRENT_EDITION == "nano" or device.type != "cpu":
        try:
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))
        except Exception:
            pass
        return 1
        
    num_cores = os.cpu_count() or 4
    
    if num_cores <= 4:
        # Crappy CPU: use max(1, cores - 1) threads, 1 worker
        torch.set_num_threads(max(1, num_cores - 1))
        return 1
    elif num_cores <= 8:
        # Mid-range CPU: use 4 threads, 1 worker
        torch.set_num_threads(4)
        return 1
    elif num_cores <= 16:
        # Good CPU (e.g. 12-16 cores): 2 parallel tile workers, each with 4 threads
        torch.set_num_threads(4)
        return 2
    else:
        # Beast CPU (e.g. 24, 32, 64 cores!): parallel workers scale with cores up to 7
        workers = min(7, max(3, num_cores // 4))
        torch.set_num_threads(4)
        return workers

# ============================================================================
# Model cache
# ============================================================================

_CORRECTION_CACHE:  Optional[tuple] = None
_STAR_REDUCE_CACHE: Optional[tuple] = None
_SHARPEN_CACHE:     Optional[tuple] = None


def _checkpoint_args(ckpt: dict) -> dict:
    """Return training arguments saved by the Aesthetics trainers."""
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if isinstance(args, argparse.Namespace):
        return vars(args)
    return args if isinstance(args, dict) else {}


def _finalize_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Move an inference model to its backend and use FP16 on native GPUs."""
    model.eval().to(device)
    if device.type in ("cuda", "mps", "xpu"):
        model.half()
    return model


def _load_weights_strict(model: nn.Module, state_dict: dict, label: str) -> None:
    """Never silently run a partial/randomly initialised neural network."""
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{label} checkpoint is incompatible with its model architecture: {exc}") from exc


def _load_correction_model(path: str, device: torch.device, mode="classic") -> nn.Module:
    global _CORRECTION_CACHE
    if (_CORRECTION_CACHE and _CORRECTION_CACHE[0] == path
            and str(_CORRECTION_CACHE[2]) == str(device)):
        return _CORRECTION_CACHE[1]
    ckpt = _load_pth(path)
    sd   = extract_state_dict(ckpt)
    cleaned_sd = _clean_state_dict(sd)
    mc   = ckpt.get("config", {}).get("model", {})
    if _CURRENT_EDITION == "pro" and mode == "aesthetics":
        width = cleaned_sd["intro.weight"].shape[0] if "intro.weight" in cleaned_sd else mc.get("width", 32)
        in_channels = cleaned_sd["intro.weight"].shape[1] if "intro.weight" in cleaned_sd else mc.get("img_channels", 3)
        out_channels = cleaned_sd["ending.weight"].shape[0] if "ending.weight" in cleaned_sd else 3
        model = NAFNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            middle_blk_num=mc.get("middle_blk_num", 4),
            enc_blk_nums=mc.get("enc_blk_nums", [2, 2, 2]),
            dec_blk_nums=mc.get("dec_blk_nums", [2, 2, 2])
        )
    else:
        tc   = ckpt.get("config", {}).get("training", {})
        model = StellarDirectNet(
            in_channels=mc.get("in_channels", 3),
            base_channels=mc.get("base_channels", 32),
            num_res_blocks=mc.get("num_res_blocks", 2),
            coord_conv=mc.get("coord_conv", True),
            cond_level=False,
            aggressive_correction=tc.get("aggressive_correction", False),
        )
    _load_weights_strict(model, cleaned_sd, "Correction")
    model = _finalize_model(model, device)
    _CORRECTION_CACHE = (path, model, device)
    return model


def _load_star_reduce_model(path: str, device: torch.device, mode="classic") -> nn.Module:
    global _STAR_REDUCE_CACHE
    if (_STAR_REDUCE_CACHE and _STAR_REDUCE_CACHE[0] == path
            and str(_STAR_REDUCE_CACHE[2]) == str(device)):
        return _STAR_REDUCE_CACHE[1]
    ckpt = _load_pth(path)
    sd   = extract_state_dict(ckpt)
    cleaned_sd = _clean_state_dict(sd)
    mc   = ckpt.get("config", {}).get("model", {})
    if _CURRENT_EDITION == "pro" and mode == "aesthetics":
        args = _checkpoint_args(ckpt)
        width = cleaned_sd["intro.weight"].shape[0] if "intro.weight" in cleaned_sd else mc.get("width", 32)
        in_channels = cleaned_sd["intro.weight"].shape[1] if "intro.weight" in cleaned_sd else mc.get("img_channels", 4)
        out_channels = cleaned_sd["ending.weight"].shape[0] if "ending.weight" in cleaned_sd else 3
        model = NAFNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            middle_blk_num=args.get("middle_blk_num", mc.get("middle_blk_num", 4)),
            enc_blk_nums=args.get("enc_blk_nums", mc.get("enc_blk_nums", [2, 2, 2])),
            dec_blk_nums=args.get("dec_blk_nums", mc.get("dec_blk_nums", [2, 2, 2]))
        )
    else:
        tc   = ckpt.get("config", {}).get("training", {})
        model = StellarDirectNet(
            in_channels=mc.get("in_channels", 3),
            base_channels=mc.get("base_channels", 32),
            num_res_blocks=mc.get("num_res_blocks", 2),
            coord_conv=mc.get("coord_conv", True),
            cond_level=mc.get("cond_level", True),
            aggressive_correction=tc.get("aggressive_correction", False),
        )
    _load_weights_strict(model, cleaned_sd, "Star reduction")
    model = _finalize_model(model, device)
    _STAR_REDUCE_CACHE = (path, model, device)
    return model


def _load_sharpen_model(path: str, device: torch.device, mode="classic") -> nn.Module:
    global _SHARPEN_CACHE
    if (_SHARPEN_CACHE and _SHARPEN_CACHE[0] == path
            and str(_SHARPEN_CACHE[2]) == str(device)):
        return _SHARPEN_CACHE[1]
    ckpt  = _load_pth(path)
    sd    = extract_state_dict(ckpt)
    cleaned_sd = _clean_state_dict(sd)
    if _CURRENT_EDITION == "pro" and mode == "aesthetics":
        mc = ckpt.get("config", {}).get("model", {})
        width = cleaned_sd["intro.weight"].shape[0] if "intro.weight" in cleaned_sd else mc.get("width", 32)
        in_channels = cleaned_sd["intro.weight"].shape[1] if "intro.weight" in cleaned_sd else mc.get("img_channels", 3)
        out_channels = cleaned_sd["ending.weight"].shape[0] if "ending.weight" in cleaned_sd else 3
        model = NAFNet(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            middle_blk_num=mc.get("middle_blk_num", 4),
            enc_blk_nums=mc.get("enc_blk_nums", [2, 2, 2]),
            dec_blk_nums=mc.get("dec_blk_nums", [2, 2, 2])
        )
    else:
        args  = ckpt.get("args", {})
        preset        = args.get("preset", "balanced")
        upsample_mode = args.get("upsample_mode", "bilinear")
        model = build_nafnet_lite(preset_name=preset, upsample_mode=upsample_mode, img_channels=3)
    _load_weights_strict(model, cleaned_sd, "Sharpen")
    model = _finalize_model(model, device)
    _SHARPEN_CACHE = (path, model, device)
    return model


def clear_model_cache():
    global _CORRECTION_CACHE, _STAR_REDUCE_CACHE, _SHARPEN_CACHE
    _CORRECTION_CACHE = _STAR_REDUCE_CACHE = _SHARPEN_CACHE = None

# ============================================================================
# Tiling helpers
# ============================================================================

def _pad_reflect(img: np.ndarray, pad: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img.shape[:2]
    if pad <= 0:
        return img, (h, w)
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return padded.astype(np.float32, copy=False), (h, w)


def _unpad(img: np.ndarray, orig_hw: Tuple[int, int], pad: int) -> np.ndarray:
    h, w = orig_hw
    if pad <= 0:
        return img[:h, :w, :]
    return img[pad:pad + h, pad:pad + w, :].astype(np.float32, copy=False)


def _grid_positions(length: int, tile: int, overlap: int) -> list:
    if length <= tile:
        return [0]
    stride = tile - overlap
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _pad_tile(tile: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = tile.shape[:2]
    if h == tile_size and w == tile_size:
        return tile
    mode = "reflect" if h > 1 and w > 1 else "edge"
    return np.pad(tile, ((0, tile_size - h), (0, tile_size - w), (0, 0)), mode=mode)


def _cosine_weights(tile: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((tile, tile, 1), dtype=np.float32)
    coords = np.arange(tile, dtype=np.float32)
    dist   = np.minimum(coords, coords[::-1])
    ramp   = np.clip(dist / float(overlap), 0.0, 1.0)
    w1d    = (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
    w1d    = np.clip(w1d, 1e-3, 1.0)
    return (w1d[:, None] * w1d[None, :])[..., None]


def _tent_weights(tile: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((tile, tile, 1), dtype=np.float32)
    coords = torch.linspace(-1.0, 1.0, steps=tile)
    ramp   = torch.clamp(1.0 - torch.abs(coords), min=1e-3)
    w      = torch.outer(ramp, ramp)
    w      = (w / w.max()).numpy()[..., None].astype(np.float32)
    return w


# Brightness normalisation for correction / star_reduce (models are brightness-sensitive)
_TARGET_MEDIAN = 0.10


def _normalize_tile(patch: np.ndarray) -> Tuple[np.ndarray, float]:
    # Ensure memory contiguity to prevent CPU cache misses when slicing from giant images
    patch_cont = np.ascontiguousarray(patch)
    tile_median = max(float(np.median(patch_cont[::8, ::8, :])), 1e-3)
    scale = min(_TARGET_MEDIAN / tile_median, _TARGET_MEDIAN / 1e-3)
    return np.clip(patch_cont * scale, 0.0, 1.0).astype(np.float32, copy=False), scale


_CIRCULAR_KERNEL_CACHE = {}

def _get_circular_kernel(size: int, device) -> torch.Tensor:
    # Generates a 2D circular/disk structuring element of given size (with static caching)
    global _CIRCULAR_KERNEL_CACHE
    key = (size, str(device))
    if key in _CIRCULAR_KERNEL_CACHE:
        return _CIRCULAR_KERNEL_CACHE[key]
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    dist = x**2 + y**2
    kernel = (dist <= 1.0).float().to(device)
    _CIRCULAR_KERNEL_CACHE[key] = kernel
    return kernel


def _erode_circular(img: torch.Tensor, size: int) -> torch.Tensor:
    kernel = _get_circular_kernel(size, img.device)
    K = size
    padding = K // 2
    B, C, H, W = img.shape
    
    # Process channel-wise using unfolding on flattened batch & channel dimensions
    x = img.view(B * C, 1, H, W)
    patches = F.unfold(x, kernel_size=K, padding=padding) # (B*C, K*K, H*W)
    
    k_flat = kernel.view(-1, 1) # (K*K, 1)
    large_val = 999.0
    masked_patches = torch.where(k_flat == 1.0, patches, torch.tensor(large_val, dtype=patches.dtype, device=patches.device))
    
    min_vals, _ = torch.min(masked_patches, dim=1) # (B*C, H*W)
    return min_vals.view(B, C, H, W)


def _preserve_dark_background(pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Remove per-channel DC drift measured on the darker half of each tile.

    Star reduction is a local operation; a uniform RGB offset in dark pixels is
    an unwanted model bias.  Estimating it per image/channel also prevents the
    temporary unlinked MTF stretch from magnifying a tiny colour cast.
    """
    pred_f = pred.float()
    source_f = source.float()
    luminance = source_f.mean(dim=1, keepdim=True)
    threshold = luminance.mean(dim=(2, 3), keepdim=True)
    dark_sample = (luminance <= threshold).to(source_f.dtype)
    denom = dark_sample.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    channel_drift = ((pred_f - source_f) * dark_sample).sum(dim=(2, 3), keepdim=True) / denom
    # Fade the correction out before reaching bright stars/nebulosity, otherwise
    # restoring the sky offset would also weaken the intended star reduction.
    safe_threshold = threshold.clamp_min(1e-4)
    background_blend = ((1.5 * safe_threshold - luminance) / (0.5 * safe_threshold)).clamp(0.0, 1.0)
    return (pred_f - channel_drift * background_blend).to(pred.dtype)


def _classic_masked_star_reduction(source: torch.Tensor, mask_reference: torch.Tensor,
                                   star_level: float) -> torch.Tensor:
    """Classic star reduction with a neural star mask and bounded round cores.

    The Classic model supplies the semantic star mask through its prediction
    delta.  Reduction itself is applied to luminance with one circular erosion
    only: repeated RGB erosions at high levels were able to collapse stellar
    cores to black.  A level-dependent luminance floor keeps the result round,
    sharp and colour-preserving while increasing reduction progressively.
    """
    diff = torch.abs(mask_reference.float() - source.float())
    diff_max = torch.max(diff, dim=1, keepdim=True)[0]
    int_level = min(max(int(math.floor(star_level)), 1), 7)
    # disk size, mask threshold, mask slope, mask smoothing, reduction blend,
    # and the minimum retained fraction of a stellar core's luminance.
    settings = {
        1: (3, 0.012, 18.0, 3, 0.30, 0.88),
        2: (3, 0.010, 24.0, 3, 0.40, 0.82),
        3: (3, 0.008, 32.0, 3, 0.50, 0.76),
        4: (5, 0.007, 42.0, 3, 0.60, 0.70),
        5: (5, 0.006, 56.0, 5, 0.68, 0.64),
        6: (5, 0.005, 72.0, 5, 0.76, 0.58),
        7: (7, 0.004, 90.0, 5, 0.82, 0.52),
    }
    size, thresh, slope, blur_size, blend, core_floor = settings[int_level]
    mask = torch.clamp((diff_max - thresh) * slope, 0.0, 1.0)
    mask = F.max_pool2d(mask, kernel_size=blur_size, stride=1, padding=blur_size // 2)
    mask = F.avg_pool2d(mask, kernel_size=blur_size, stride=1, padding=blur_size // 2)

    source_f = source.float()
    # Work on luminance so coloured stars are reduced as coloured stars, not
    # independently eroded RGB channels.
    luminance = source_f.mean(dim=1, keepdim=True)
    eroded_luminance = _erode_circular(luminance, size=size)
    bounded_luminance = torch.maximum(eroded_luminance, luminance * core_floor)
    ratio = bounded_luminance / luminance.clamp_min(1e-5)
    rounded_target = source_f * ratio

    if star_level < 1.0:
        blend *= max(0.0, float(star_level))
    return (source_f + blend * mask * (rounded_target - source_f)).to(source.dtype)


def _clean_classic_nebula_deblur(pred: torch.Tensor, source: torch.Tensor,
                                 strength: float) -> torch.Tensor:
    """Protect point sources and reject isolated deblur artefacts in Classic.

    The Classic deblur checkpoint is effective on broad nebulosity, but its
    residual can become gritty when amplified.  This soft mask attenuates the
    residual on compact local peaks (stars) and suppresses isolated residual
    pixels while retaining spatially coherent nebula detail.  Protection grows
    smoothly with the user strength above 1.0.
    """
    source_f = source.float()
    delta = pred.float() - source_f
    luminance = source_f.mean(dim=1, keepdim=True)
    local_background = F.avg_pool2d(luminance, kernel_size=9, stride=1, padding=4)
    peak_contrast = ((luminance - local_background - 0.008) / 0.075).clamp(0.0, 1.0)
    peak_brightness = ((luminance - 0.16) / 0.30).clamp(0.0, 1.0)
    star_mask = F.max_pool2d(
        peak_contrast * peak_brightness, kernel_size=5, stride=1, padding=2
    )

    residual_energy = delta.abs().mean(dim=1, keepdim=True)
    coherent_energy = F.avg_pool2d(residual_energy, kernel_size=5, stride=1, padding=2)
    coherence = (coherent_energy / residual_energy.clamp_min(1e-4)).clamp(0.0, 1.0)
    high_strength = min(max((float(strength) - 1.0) / 2.0, 0.0), 1.0)
    star_keep = 1.0 - star_mask * (0.45 + 0.45 * high_strength)
    artefact_keep = 1.0 - (1.0 - coherence) * (0.10 + 0.35 * high_strength)
    return (source_f + delta * star_keep * artefact_keep).to(pred.dtype)


def _run_tiled_correction_or_starreduce(
    model: nn.Module,
    device: torch.device,
    img_rgb01: np.ndarray,
    tile: int,
    overlap: int,
    pad: int,
    star_level: Optional[int],   # None = correction, int = star reduce
    progress_cb: Optional[Callable],
    label: str,
    mode: str = "classic",
    live_update_cb: Optional[Callable[[np.ndarray], None]] = None,
    batch_size_cfg: str = "Auto",
) -> np.ndarray:
    """Cosine-blend tiling with per-tile brightness normalisation."""
    x_padded, orig_hw = _pad_reflect(img_rgb01, pad)
    H, W = x_padded.shape[:2]
    ys   = _grid_positions(H, tile, overlap)
    xs   = _grid_positions(W, tile, overlap)
    total = len(ys) * len(xs)
    w2    = _cosine_weights(tile, overlap)
    # Full-frame preview reconstruction is O(image_pixels), so cap it to about
    # ten updates per stage instead of doing it every two tiles.
    live_update_interval = max(1, math.ceil(total / 10))

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc   = np.zeros((H, W, 1), dtype=np.float32)

    lv_tensor = None
    if star_level is not None and star_level > 0:
        lv_tensor = torch.tensor([(star_level - 1.0) / 6.0], dtype=torch.float32).to(device)

    # Resolve batch size on GPU
    active_batch_size = _resolve_inference_batch_size(total, tile, device, batch_size_cfg)
    print(f"SyQon Parallax [{label}]: mode={mode}, device={device}, batch_size_cfg={batch_size_cfg} -> active_batch_size={active_batch_size}")

    num_workers = _configure_cpu_performance(device)

    if num_workers > 1:
        lock = threading.Lock()
        done_counter = 0

        def process_tile(y0, x0):
            nonlocal done_counter
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ph, pw  = y1 - y0, x1 - x0
            patch   = x_padded[y0:y1, x0:x1, :]
            patch_full = _pad_tile(patch, tile)

            if mode == "aesthetics":
                patch_norm = patch_full
                scale = 1.0
            else:
                patch_norm, scale = _normalize_tile(patch_full)
            t = torch.from_numpy(patch_norm.transpose(2, 0, 1)[None]).float().to(device).contiguous()
            is_half = next(model.parameters()).dtype == torch.float16
            if is_half:
                t = t.half()

            with torch.inference_mode():
                if _CURRENT_EDITION == "pro" and mode == "aesthetics":
                    if star_level is not None:
                        b_curr = t.shape[0]
                        lvl_norm = (float(star_level) - 1.0) / 6.0
                        lvl_channel = torch.full((b_curr, 1, t.shape[2], t.shape[3]), lvl_norm, dtype=t.dtype, device=t.device)
                        t_4ch = torch.cat([t, lvl_channel], dim=1)
                        pred = model(t_4ch)
                        pred = _preserve_dark_background(pred, t)
                    else:
                        pred = model(t)
                        
                    # Saturated and mid-tone core preservation: fix magenta/purple cores introduced by the model
                    t_max = torch.max(t, dim=1, keepdim=True)[0]
                    sat_mask = t_max > 0.70
                    t_r, t_g, t_b = t[:, 0:1, :, :], t[:, 1:2, :, :], t[:, 2:3, :, :]
                    t_rb_min = torch.min(t_r, t_b)
                    input_not_magenta = (t_rb_min - t_g) < 0.08
                    
                    pred_r, pred_g, pred_b = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]
                    pred_rb_min = torch.min(pred_r, pred_b)
                    magenta_mask = (pred_rb_min - pred_g) > 0.08
                    corrected_g = torch.where(sat_mask & input_not_magenta & magenta_mask, pred_rb_min, pred_g)
                    pred = torch.cat([pred_r, corrected_g, pred_b], dim=1)
                else:
                    if star_level is not None and star_level > 0:
                        # Classic: the model identifies stars; bounded circular
                        # luminance reduction keeps high-level cores from blackening.
                        safe_lv = torch.tensor([5.0 / 10.0], dtype=torch.float32).to(device)
                        mask_reference = model(t, safe_lv.half() if is_half else safe_lv)
                        pred = _classic_masked_star_reduction(t, mask_reference, float(star_level))
                    elif lv_tensor is not None:
                        pred = model(t, lv_tensor)
                    else:
                        pred = model(t)

            pred_np = pred[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)
            pred_tile = np.clip(pred_np / scale, 0.0, 1.0).astype(np.float32)

            wlocal = w2[:ph, :pw, :]
            with lock:
                out_acc[y0:y1, x0:x1, :] += pred_tile[:ph, :pw, :] * wlocal
                w_acc[y0:y1, x0:x1, :] += wlocal
                done_counter += 1
                if callable(progress_cb):
                    progress_cb(done_counter, total, f"{label} tile {done_counter}/{total}")
                if live_update_cb is not None and (done_counter % live_update_interval == 0 or done_counter == total):
                    mask = w_acc > 1e-5
                    interm = np.where(mask, out_acc / (w_acc + 1e-8), x_padded[:, :, :3])
                    interm_unpadded = _unpad(interm, orig_hw, pad)
                    live_update_cb(interm_unpadded)

        tasks = []
        for y0 in ys:
            for x0 in xs:
                tasks.append((y0, x0))
                
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(lambda p: process_tile(*p), tasks)

    else:
        coords = []
        for y0 in ys:
            for x0 in xs:
                coords.append((y0, x0))

        # Parallelize CPU-bound tile extraction, padding, and normalisation across all available CPU cores
        def prepare_tile(y0_x0):
            y0, x0 = y0_x0
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ph, pw  = y1 - y0, x1 - x0
            patch   = x_padded[y0:y1, x0:x1, :]
            patch_full = _pad_tile(patch, tile)
            
            if mode == "aesthetics":
                patch_norm = patch_full
                scale = 1.0
            else:
                patch_norm, scale = _normalize_tile(patch_full)
                
            t_tile = torch.from_numpy(patch_norm.transpose(2, 0, 1)[None]).float()
            return t_tile, scale, (y0, x0, y1, x1, ph, pw)

        done = 0
        # Prepare only the next inference batch.  The previous implementation
        # materialised every 512x512 float tile at once, duplicating hundreds of
        # MB on large astro images before inference even started.
        prepare_workers = min(os.cpu_count() or 4, max(1, active_batch_size))
        def iter_prepared_batches():
            with ThreadPoolExecutor(max_workers=prepare_workers) as executor:
                for i in range(0, len(coords), active_batch_size):
                    batch_coords = coords[i : i + active_batch_size]
                    yield list(executor.map(prepare_tile, batch_coords))

        for batch_data in iter_prepared_batches():
            
            batch_tensors = [r[0] for r in batch_data]
            batch_scales  = [r[1] for r in batch_data]
            batch_meta    = [r[2] for r in batch_data]
            
            t_batch = torch.cat(batch_tensors, dim=0).to(device).contiguous()
            is_half = next(model.parameters()).dtype == torch.float16
            if is_half:
                t_batch = t_batch.half()
            
            with torch.inference_mode():
                if _CURRENT_EDITION == "pro" and mode == "aesthetics":
                    if star_level is not None:
                        b_curr = t_batch.shape[0]
                        lvl_norm = (float(star_level) - 1.0) / 6.0
                        lvl_channel = torch.full((b_curr, 1, t_batch.shape[2], t_batch.shape[3]), lvl_norm, dtype=t_batch.dtype, device=t_batch.device)
                        t_4ch = torch.cat([t_batch, lvl_channel], dim=1)
                        pred = model(t_4ch)
                        pred = _preserve_dark_background(pred, t_batch)
                    else:
                        pred = model(t_batch)
                        
                    # Saturated and mid-tone core preservation: fix magenta/purple cores introduced by the model
                    t_max = torch.max(t_batch, dim=1, keepdim=True)[0]
                    sat_mask = t_max > 0.70
                    t_r, t_g, t_b = t_batch[:, 0:1, :, :], t_batch[:, 1:2, :, :], t_batch[:, 2:3, :, :]
                    t_rb_min = torch.min(t_r, t_b)
                    input_not_magenta = (t_rb_min - t_g) < 0.08
                    
                    pred_r, pred_g, pred_b = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]
                    pred_rb_min = torch.min(pred_r, pred_b)
                    magenta_mask = (pred_rb_min - pred_g) > 0.08
                    corrected_g = torch.where(sat_mask & input_not_magenta & magenta_mask, pred_rb_min, pred_g)
                    pred = torch.cat([pred_r, corrected_g, pred_b], dim=1)
                else:
                    # Nano (mini) sequential fallback (runs with active_batch_size = 1)
                    t_single = t_batch
                    if star_level is not None and star_level > 0:
                        safe_lv = torch.tensor([5.0 / 10.0], dtype=torch.float32).to(device)
                        mask_reference = model(t_single, safe_lv.half() if is_half else safe_lv)
                        pred = _classic_masked_star_reduction(t_single, mask_reference, float(star_level))
                    elif lv_tensor is not None:
                        pred = model(t_single, lv_tensor.half() if is_half else lv_tensor)
                    else:
                        pred = model(t_single)

            pred_np_batch = pred.float().cpu().numpy()

            # Unpack predictions from the batch
            for idx, (y0, x0, y1, x1, ph, pw) in enumerate(batch_meta):
                scale = batch_scales[idx]
                pred_tile = pred_np_batch[idx].transpose(1, 2, 0)
                pred_tile = np.clip(pred_tile / scale, 0.0, 1.0)

                wlocal = w2[:ph, :pw, :]
                out_acc[y0:y1, x0:x1, :] += pred_tile[:ph, :pw, :] * wlocal
                w_acc  [y0:y1, x0:x1, :] += wlocal

                done += 1
                if callable(progress_cb):
                    progress_cb(done, total, f"{label} tile {done}/{total}")
                if live_update_cb is not None and (done % live_update_interval == 0 or done == total):
                    mask = w_acc > 1e-5
                    interm = np.where(mask, out_acc / (w_acc + 1e-8), x_padded[:, :, :3])
                    interm_unpadded = _unpad(interm, orig_hw, pad)
                    live_update_cb(interm_unpadded)


    # Empty GPU cache once at the end of the entire tiled process to avoid thread stuttering
    _empty_device_cache(device)

    np.maximum(w_acc, 1e-8, out=w_acc)
    result = np.clip(out_acc / w_acc, 0.0, 1.0)
    return _unpad(result, orig_hw, pad)


def _run_tiled_sharpen(
    model: nn.Module,
    device: torch.device,
    img_rgb01: np.ndarray,
    alpha: float,
    tile: int,
    overlap: int,
    pad: int,
    progress_cb: Optional[Callable],
    live_update_cb: Optional[Callable[[np.ndarray], None]] = None,
    batch_size_cfg: str = "Auto",
    mode: str = "classic",
    classic_strength: float = 1.0,
) -> np.ndarray:
    """Tent-window tiling with reflect-pad. NO brightness normalisation."""
    x_padded, orig_hw = _pad_reflect(img_rgb01, pad)
    H, W  = x_padded.shape[:2]
    ys    = _grid_positions(H, tile, overlap)
    xs    = _grid_positions(W, tile, overlap)
    total = len(ys) * len(xs)
    w2    = _tent_weights(tile, overlap)
    live_update_interval = max(1, math.ceil(total / 10))

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_sum   = np.zeros((H, W, 1), dtype=np.float32)

    # Resolve batch size on GPU
    active_batch_size = _resolve_inference_batch_size(total, tile, device, batch_size_cfg)
    print(f"SyQon Parallax [Sharpen]: mode={mode}, device={device}, batch_size_cfg={batch_size_cfg} -> active_batch_size={active_batch_size}")

    num_workers = _configure_cpu_performance(device)

    if num_workers > 1:
        lock = threading.Lock()
        done_counter = 0

        def process_tile(y0, x0):
            nonlocal done_counter
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ph, pw  = y1 - y0, x1 - x0
            patch   = x_padded[y0:y1, x0:x1, :]
            patch_full = _pad_tile(patch, tile)
            t = torch.from_numpy(patch_full.transpose(2, 0, 1)[None]).float().to(device).contiguous()
            is_half = next(model.parameters()).dtype == torch.float16
            if is_half:
                t = t.half()

            with torch.inference_mode():
                pred = model(t)

            if _CURRENT_EDITION == "pro" and mode == "classic":
                pred = _clean_classic_nebula_deblur(pred, t, classic_strength)

            if mode == "aesthetics":
                # Saturated and mid-tone core preservation: fix magenta/purple cores introduced by the model
                t_max = torch.max(t, dim=1, keepdim=True)[0]
                sat_mask = t_max > 0.70
                t_r, t_g, t_b = t[:, 0:1, :, :], t[:, 1:2, :, :], t[:, 2:3, :, :]
                t_rb_min = torch.min(t_r, t_b)
                input_not_magenta = (t_rb_min - t_g) < 0.08
                
                pred_r, pred_g, pred_b = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]
                pred_rb_min = torch.min(pred_r, pred_b)
                magenta_mask = (pred_rb_min - pred_g) > 0.08
                corrected_g = torch.where(sat_mask & input_not_magenta & magenta_mask, pred_rb_min, pred_g)
                pred = torch.cat([pred_r, corrected_g, pred_b], dim=1)

            pred_np = pred[0].float().cpu().numpy().transpose(1, 2, 0)
            wlocal = w2[:ph, :pw, :]
            with lock:
                out_acc[y0:y1, x0:x1, :] += pred_np[:ph, :pw, :] * wlocal
                w_sum[y0:y1, x0:x1, :] += wlocal
                done_counter += 1
                if callable(progress_cb):
                    progress_cb(done_counter, total, f"[Sharpen] tile {done_counter}/{total}")
                if live_update_cb is not None and (done_counter % live_update_interval == 0 or done_counter == total):
                    mask = w_sum > 1e-5
                    interm = np.where(mask, out_acc / (w_sum + 1e-8), x_padded[:, :, :3])
                    interm_unpadded = _unpad(interm, orig_hw, pad)
                    live_update_cb(interm_unpadded)

        tasks = []
        for y0 in ys:
            for x0 in xs:
                tasks.append((y0, x0))
                
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(lambda p: process_tile(*p), tasks)

    else:
        coords = []
        for y0 in ys:
            for x0 in xs:
                coords.append((y0, x0))

        # Parallelize CPU-bound tile extraction and padding across all available CPU cores
        def prepare_tile(y0_x0):
            y0, x0 = y0_x0
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ph, pw  = y1 - y0, x1 - x0
            patch   = x_padded[y0:y1, x0:x1, :]
            patch_full = _pad_tile(patch, tile)
            t_tile = torch.from_numpy(patch_full.transpose(2, 0, 1)[None]).float()
            return t_tile, (y0, x0, y1, x1, ph, pw)

        done = 0
        prepare_workers = min(os.cpu_count() or 4, max(1, active_batch_size))
        def iter_prepared_batches():
            with ThreadPoolExecutor(max_workers=prepare_workers) as executor:
                for i in range(0, len(coords), active_batch_size):
                    batch_coords = coords[i : i + active_batch_size]
                    yield list(executor.map(prepare_tile, batch_coords))

        for batch_data in iter_prepared_batches():
            
            batch_tensors = [r[0] for r in batch_data]
            batch_meta    = [r[1] for r in batch_data]
            
            t_batch = torch.cat(batch_tensors, dim=0).to(device).contiguous()
            is_half = next(model.parameters()).dtype == torch.float16
            if is_half:
                t_batch = t_batch.half()
            
            with torch.inference_mode():
                pred = model(t_batch)

            if _CURRENT_EDITION == "pro" and mode == "classic":
                pred = _clean_classic_nebula_deblur(pred, t_batch, classic_strength)

            if mode == "aesthetics":
                # Saturated and mid-tone core preservation: fix magenta/purple cores introduced by the model
                t_max = torch.max(t_batch, dim=1, keepdim=True)[0]
                sat_mask = t_max > 0.70
                t_r, t_g, t_b = t_batch[:, 0:1, :, :], t_batch[:, 1:2, :, :], t_batch[:, 2:3, :, :]
                t_rb_min = torch.min(t_r, t_b)
                input_not_magenta = (t_rb_min - t_g) < 0.08
                
                pred_r, pred_g, pred_b = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]
                pred_rb_min = torch.min(pred_r, pred_b)
                magenta_mask = (pred_rb_min - pred_g) > 0.08
                corrected_g = torch.where(sat_mask & input_not_magenta & magenta_mask, pred_rb_min, pred_g)
                pred = torch.cat([pred_r, corrected_g, pred_b], dim=1)

            pred_np_batch = pred.float().cpu().numpy()

            # Unpack predictions from the batch
            for idx, (y0, x0, y1, x1, ph, pw) in enumerate(batch_meta):
                pred_tile = pred_np_batch[idx].transpose(1, 2, 0)
                wlocal = w2[:ph, :pw, :]
                out_acc[y0:y1, x0:x1, :] += pred_tile[:ph, :pw, :] * wlocal
                w_sum  [y0:y1, x0:x1, :] += wlocal

                done += 1
                if callable(progress_cb):
                    progress_cb(done, total, f"[Sharpen] tile {done}/{total}")

            if live_update_cb is not None and (done % live_update_interval == 0 or done == total):
                mask = w_sum > 1e-5
                interm = np.where(mask, out_acc / (w_sum + 1e-8), x_padded[:, :, :3])
                interm_unpadded = _unpad(interm, orig_hw, pad)
                live_update_cb(interm_unpadded)


    # Empty GPU cache once at the end of the entire tiled process to avoid thread stuttering
    _empty_device_cache(device)

    np.maximum(w_sum, 1e-8, out=w_sum)
    reconstructed = np.clip(out_acc / w_sum, 0.0, 1.0)
    reconstructed = _unpad(reconstructed, orig_hw, pad)
    result = img_rgb01 + alpha * (reconstructed - img_rgb01)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

# ============================================================================
# High-level process_image() — three-stage pipeline
# ============================================================================

def process_image(
    img_hwc3: np.ndarray,
    *,
    correct:        bool  = True,
    star_level:     int   = 3,
    sharpen_alpha:  float = 1.0,
    tile:           int   = 512,
    overlap:        int   = 64,
    pad:            int   = 96,
    use_mtf:        bool  = True,
    mtf_target:     float = 0.15,
    linked_stretch: bool  = False,
    use_gpu:        bool  = True,
    correction_path:  str = "",
    star_reduce_path: str = "",
    sharpen_path:     str = "",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    mode:           str = "classic",
    live_update_cb: Optional[Callable[[np.ndarray], None]] = None,
    batch_size_cfg: str = "Auto",
    allow_large_live_preview: bool = False,
) -> np.ndarray:
    """
    Full three-stage Parallax pipeline on a HWC float32 [0,1] image with robust GPU fallback.
    Returns HWC float32 [0,1].
    """
    device = pick_device(use_gpu)

    try:
        return _process_image_impl(
            img_hwc3,
            correct=correct,
            star_level=star_level,
            sharpen_alpha=sharpen_alpha,
            tile=tile,
            overlap=overlap,
            pad=pad,
            use_mtf=use_mtf,
            mtf_target=mtf_target,
            linked_stretch=linked_stretch,
            device=device,
            correction_path=correction_path,
            star_reduce_path=star_reduce_path,
            sharpen_path=sharpen_path,
            progress_cb=progress_cb,
            mode=mode,
            live_update_cb=live_update_cb,
            batch_size_cfg=batch_size_cfg,
            allow_large_live_preview=allow_large_live_preview,
        )
    except Exception as exc:
        if device.type == "cpu":
            raise exc
        
        print(f"Warning: Processing failed on GPU {device} due to: {exc}. Retrying on CPU...")
        if callable(progress_cb):
            progress_cb(0, 1, f"GPU {device} failed; retrying on CPU: {exc}")
        
        global _CORRECTION_CACHE, _STAR_REDUCE_CACHE, _SHARPEN_CACHE
        _CORRECTION_CACHE = None
        _STAR_REDUCE_CACHE = None
        _SHARPEN_CACHE = None
        _empty_device_cache(device)
            
        return _process_image_impl(
            img_hwc3,
            correct=correct,
            star_level=star_level,
            sharpen_alpha=sharpen_alpha,
            tile=tile,
            overlap=overlap,
            pad=pad,
            use_mtf=use_mtf,
            mtf_target=mtf_target,
            linked_stretch=linked_stretch,
            device=torch.device("cpu"),
            correction_path=correction_path,
            star_reduce_path=star_reduce_path,
            sharpen_path=sharpen_path,
            progress_cb=progress_cb,
            mode=mode,
            live_update_cb=live_update_cb,
            batch_size_cfg=batch_size_cfg,
            allow_large_live_preview=allow_large_live_preview,
        )


def _process_image_impl(
    img_hwc3: np.ndarray,
    *,
    correct:        bool  = True,
    star_level:     int   = 3,
    sharpen_alpha:  float = 1.0,
    tile:           int   = 512,
    overlap:        int   = 64,
    pad:            int   = 96,
    use_mtf:        bool  = True,
    mtf_target:     float = 0.15,
    linked_stretch: bool  = False,
    device:         torch.device,
    correction_path:  str = "",
    star_reduce_path: str = "",
    sharpen_path:     str = "",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    mode:           str = "classic",
    live_update_cb: Optional[Callable[[np.ndarray], None]] = None,
    batch_size_cfg: str = "Auto",
    allow_large_live_preview: bool = False,
) -> np.ndarray:
    def _emit(done, total, msg):
        if callable(progress_cb):
            progress_cb(done, total, msg)

    n_stages = sum([int(correct), int(star_level > 0), int(sharpen_alpha > 0)])
    if n_stages == 0:
        return img_hwc3.astype(np.float32, copy=False)

    stage_idx = 0
    current   = img_hwc3.astype(np.float32, copy=False)

    requested_tile = tile
    tile, overlap, large_image_mode = _tune_large_image_geometry(current, tile, overlap, device)
    live_updates_enabled = callable(live_update_cb) and (
        not large_image_mode or allow_large_live_preview
    )
    if large_image_mode:
        megapixels = current.shape[0] * current.shape[1] / 1_000_000.0
        geometry_note = f", tile {requested_tile}->{tile}" if tile != requested_tile else ""
        if allow_large_live_preview:
            _emit(0, 1, f"Large image ({megapixels:.1f} MP{geometry_note}): live preview kept enabled.")
        else:
            _emit(
                0, 1,
                f"The image is too large ({megapixels:.1f} MP{geometry_note}), so live preview "
                "has been disabled to preserve performance."
            )

    # MTF stretch
    mtf_params = None
    if use_mtf:
        _emit(0, 1, "Applying temporary stretch…")
        mtf_params = (_mtf_params_linked(current, targetbg=mtf_target)
                      if linked_stretch
                      else _mtf_params_unlinked(current, targetbg=mtf_target))
        current = apply_mtf_stretch(current, mtf_params)

    # Define dynamic live feedback emitter
    def _live_emit(intermediate_img, stage="correct"):
        if not callable(live_update_cb):
            return
        if use_mtf and mtf_params is not None:
            overall_unstretched = apply_mtf_inverse(intermediate_img, mtf_params)
        else:
            overall_unstretched = intermediate_img
        live_update_cb(np.clip(overall_unstretched, 0.0, 1.0).astype(np.float32, copy=False))

    # --- Sequential Pipeline: Stage 1. Aberration Correction ---
    if correct:
        stage_idx += 1
        _emit(0, 1, f"[{stage_idx}/{n_stages}] Preparing correction stage…")
        model_c = _load_correction_model(correction_path, device, mode if _CURRENT_EDITION == "pro" else "classic")

        def _prog_c(done, total, msg):
            _emit(done, total, f"[{stage_idx}/{n_stages}] Correction — {msg}")

        current = _run_tiled_correction_or_starreduce(
            model_c, device, current, tile, overlap, pad,
            star_level=None, progress_cb=_prog_c, label="Correction", mode=mode if _CURRENT_EDITION == "pro" else "classic",
            live_update_cb=(lambda img: _live_emit(img, stage="correct")) if live_updates_enabled else None,
            batch_size_cfg=batch_size_cfg
        )

    # --- Sequential Pipeline: Stage 2. Star Reduction ---
    if star_level > 0:
        stage_idx += 1
        _emit(0, 1, f"[{stage_idx}/{n_stages}] Preparing star reduction stage…")
        model_r = _load_star_reduce_model(star_reduce_path, device, mode if _CURRENT_EDITION == "pro" else "classic")

        def _prog_s(done, total, msg):
            _emit(done, total, f"[{stage_idx}/{n_stages}] Star Reduction L{star_level} — {msg}")

        current = _run_tiled_correction_or_starreduce(
            model_r, device, current, tile, overlap, pad,
            star_level=star_level, progress_cb=_prog_s, label=f"StarReduce L{star_level}", mode=mode if _CURRENT_EDITION == "pro" else "classic",
            live_update_cb=(lambda img: _live_emit(img, stage="reduce" if _CURRENT_EDITION == "pro" else "nano_reduce")) if live_updates_enabled else None,
            batch_size_cfg=batch_size_cfg
        )

    # --- Sequential Pipeline: Stage 3. Nebula Deblur (non-stellar) ---
    if sharpen_alpha > 0.0:
        stage_idx += 1
        _emit(0, 1, f"[{stage_idx}/{n_stages}] Preparing nebula deblur stage…")
        model_sh = _load_sharpen_model(sharpen_path, device, mode if _CURRENT_EDITION == "pro" else "classic")

        def _prog_sh(done, total, msg):
            _emit(done, total, f"[{stage_idx}/{n_stages}] Nebula Deblur α={sharpen_alpha:.2f} — {msg}")

        deblurred = _run_tiled_sharpen(
            model_sh, device, current, 1.0, tile, overlap, pad, _prog_sh,
            live_update_cb=(lambda img: _live_emit(img, stage="sharpen")) if live_updates_enabled else None,
            batch_size_cfg=batch_size_cfg,
            mode=mode if _CURRENT_EDITION == "pro" else "classic",
            classic_strength=sharpen_alpha,
        )

        # Blend the nebula-only deblur only after stellar operations complete.
        current = current + sharpen_alpha * (deblurred - current)

    # --- Final Step: MTF Inverse Stretch ---
    if use_mtf and mtf_params is not None:
        _emit(0, 1, "Inverting temporary stretch…")
        current = apply_mtf_inverse(current, mtf_params)

    _emit(1, 1, "Done.")
    
    # Clean GPU memory/cache to prevent VRAM leaks in production
    _empty_device_cache(device)

    return np.clip(current, 0.0, 1.0).astype(np.float32, copy=False)

# ============================================================================
# Qt worker thread
# ============================================================================

class ParallaxWorker(QObject):
    progress     = Signal(int, int, str)   # done, total, msg
    live_update  = Signal(object)          # intermediate HWC numpy array
    finished_ok  = Signal(object, object) # result_for_siril, result_hwc
    finished_err = Signal(str)

    def __init__(self, raw: np.ndarray, cfg: dict,
                 orig_dtype, scale: float, orig_was_mono: bool,
                 engine_dir: Path, flip_vertical: bool = False):
        super().__init__()
        self.raw           = raw
        self.cfg           = dict(cfg)
        self.orig_dtype    = orig_dtype
        self.scale         = scale
        self.orig_was_mono = orig_was_mono
        self.engine_dir    = engine_dir
        self.flip_vertical = flip_vertical
        self._cancel       = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            cfg = self.cfg
            roi = cfg.get("roi")
            is_roi_preview = bool(roi and cfg.get("is_preview", False))
            if roi:
                roi_x, roi_y, roi_w, roi_h = roi
                raw_roi = _slice_raw_roi(self.raw, roi, flip_vertical=self.flip_vertical)
                xrgb_input, _, _, _ = _prepare_for_inference(
                    raw_roi, flip_vertical=self.flip_vertical
                )
                xrgb = None
            else:
                xrgb, _, _, _ = _prepare_for_inference(
                    self.raw, flip_vertical=self.flip_vertical
                )
                xrgb_input = xrgb

            def _prog(done, total, msg):
                if self._cancel:
                    raise InterruptedError("Cancelled")
                self.progress.emit(int(done), int(total), str(msg))

            def _live_upd(intermediate_img):
                if not cfg.get("live_update", True):
                    return
                if self._cancel:
                    raise InterruptedError("Cancelled")
                if roi:
                    # ROI preview deliberately avoids allocating a full-size
                    # intermediate image on every update.
                    return
                else:
                    self.live_update.emit(intermediate_img)

            roi_result = process_image(
                xrgb_input,
                correct        = bool(cfg.get("correct", True)),
                star_level     = int(cfg.get("star_level", 3)),
                sharpen_alpha  = float(cfg.get("sharpen_alpha", 1.0)) if bool(cfg.get("sharpen", True)) else 0.0,
                tile           = int(cfg.get("tile_size", 512)),
                overlap        = int(cfg.get("overlap", 64)),
                pad            = int(cfg.get("pad", 96)),
                use_mtf        = bool(cfg.get("use_mtf", True)),
                mtf_target     = float(cfg.get("mtf_target", 0.15)),
                linked_stretch = bool(cfg.get("linked_stretch", False)),
                use_gpu        = bool(cfg.get("use_gpu", True)),
                correction_path  = str(_correction_path(cfg.get("mode", "classic"))),
                star_reduce_path = str(_star_reduce_path(cfg.get("mode", "classic"))),
                sharpen_path     = str(_sharpen_path(cfg.get("mode", "classic"))),
                progress_cb    = _prog,
                mode           = cfg.get("mode", "classic"),
                live_update_cb = _live_upd,
                batch_size_cfg = cfg.get("batch_size", "Auto"),
                allow_large_live_preview=bool(cfg.get("allow_large_live_preview", False)),
            )
            
            if is_roi_preview:
                # The GUI composites this small crop over its cached original;
                # do not allocate a giant HWC result or unused Siril buffer.
                self.finished_ok.emit(None, roi_result)
                return
            elif roi:
                # Defensive path for programmatic non-preview ROI use.
                xrgb, _, _, _ = _prepare_for_inference(
                    self.raw, flip_vertical=self.flip_vertical
                )
                result_hwc = xrgb.copy()
                result_hwc[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w, :] = roi_result
            else:
                result_hwc = roi_result

            # Convert back to Siril layout (apply inverse flip only to the FITS data for Siril)
            siril_hwc = result_hwc.copy()
            if self.flip_vertical:
                siril_hwc = np.flip(siril_hwc, axis=0)

            if self.orig_was_mono:
                result_2d = siril_hwc.mean(axis=2)
                result_for_siril = _restore_dtype(
                    result_2d[np.newaxis, ...], self.orig_dtype, self.scale
                )
            else:
                result_for_siril = _restore_dtype(
                    siril_hwc.transpose(2, 0, 1), self.orig_dtype, self.scale
                )

            # result_hwc is kept oriented "top-down" (flipped) for GUI display consistency
            self.finished_ok.emit(result_for_siril, result_hwc)

        except InterruptedError:
            self.finished_err.emit("__cancelled__")
        except Exception as exc:
            import traceback
            self.finished_err.emit(f"{exc}\n{traceback.format_exc()}")

# ============================================================================
# Helpers: model install widget row
# ============================================================================

def _make_model_row(parent, label: str, path: Path, remove_cb):
    """Returns (row_widget, status_label) — status updates when path changes."""
    col = QWidget(parent)
    col.setObjectName("modelRow")
    hlay = QHBoxLayout(col)
    hlay.setContentsMargins(9, 5, 7, 5)
    hlay.setSpacing(6)

    installed = path.exists() and path.is_file()
    color = '#30d158' if installed else '#ff453a'
    status_text = 'Installed' if installed else 'Missing'

    status = QLabel(col)
    status.setText(f"<span style='color:{color}; font-size: 11px;'>●</span> <b>{label}:</b> <span style='color:{color}; font-size: 11px;'>{status_text}</span>")
    status.setStyleSheet("color:#ffffff; font-size:11px;")
    
    btn_r = QPushButton("Remove", col)
    btn_r.setFixedWidth(64)
    btn_r.setFixedHeight(24)
    btn_r.setStyleSheet("QPushButton { font-size: 9px; padding: 0px 5px; margin: 0px; min-height: 20px; font-weight: normal; color:#94A3B8; }")
    btn_r.clicked.connect(remove_cb)
    
    hlay.addWidget(status, 1)
    hlay.addWidget(btn_r)
    return col, status

from PySide6.QtCore import QRect

def _numpy_to_qimage(arr_rgb: np.ndarray, apply_stretch: bool, stretch_target: float, linked: bool, mtf_params: dict = None) -> QImage:
    """Converts a float32 HWC RGB [0,1] numpy array to a QImage, with optional scaling and MTF stretch."""
    preview_arr = arr_rgb.copy()

    if apply_stretch:
        if mtf_params is None:
            mtf_params = (_mtf_params_linked(preview_arr, targetbg=stretch_target)
                          if linked
                          else _mtf_params_unlinked(preview_arr, targetbg=stretch_target))
        preview_arr = apply_mtf_stretch(preview_arr, mtf_params)

    img_8bit = (np.clip(preview_arr * 255.0, 0, 255)).astype(np.uint8)
    ph, pw, _ = img_8bit.shape
    
    qimg = QImage(img_8bit.data, pw, ph, 3 * pw, QImage.Format.Format_RGB888)
    return qimg.copy()


from PySide6.QtCore import QRect, QPoint

class BeforeAfterPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_before = None # QImage
        self.img_after = None  # QImage
        self.split_ratio = 0.5 # 0.0 to 1.0
        self.zoom_factor = 1.0 # 1.0 to 10.0
        self.pan_offset = QPoint(0, 0)
        self.dragging_split = False
        self.dragging_pan = False
        self.last_mouse_pos = None
        self.show_split = True
        
        # ROI selection variables
        self.roi_rect = None # QRect in image coordinates
        self.is_selecting_roi = False
        self.roi_start = None # QPoint in widget coordinates
        self.roi_end = None # QPoint in widget coordinates
        self.roi_changed_cb = None
        
        # ROI dragging variables
        self.dragging_roi = False
        self.roi_drag_start_rect = None
        self.roi_drag_start_img_pos = None
        
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_images(self, before: QImage, after: QImage = None, reset_zoom: bool = False):
        dimensions_changed = True
        if self.img_before and before:
            if self.img_before.width() == before.width() and self.img_before.height() == before.height():
                dimensions_changed = False
        
        self.img_before = before
        self.img_after = after
        
        if reset_zoom or dimensions_changed:
            self.zoom_factor = 1.0
            self.pan_offset = QPoint(0, 0)
            self.roi_rect = None
            if self.roi_changed_cb:
                self.roi_changed_cb()
        self.update()

    def set_zoom_fit(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def set_zoom_100(self):
        if not self.img_before:
            return
        w = self.width()
        h = self.height()
        img_w = self.img_before.width()
        img_h = self.img_before.height()
        scale = min(w / img_w, h / img_h)
        if scale > 0:
            self.zoom_factor = max(1.0, 1.0 / scale)
        else:
            self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def set_zoom_200(self):
        if not self.img_before:
            return
        w = self.width()
        h = self.height()
        img_w = self.img_before.width()
        img_h = self.img_before.height()
        scale = min(w / img_w, h / img_h)
        if scale > 0:
            self.zoom_factor = max(1.0, 2.0 / scale)
        else:
            self.zoom_factor = 2.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def widget_to_image_coords(self, point: QPoint) -> QPoint:
        if not self.img_before:
            return QPoint(0, 0)
        w = self.width()
        h = self.height()
        img_w = self.img_before.width()
        img_h = self.img_before.height()
        scale = min(w / img_w, h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        zw = int(draw_w * self.zoom_factor)
        zh = int(draw_h * self.zoom_factor)
        zdx = (w - zw) // 2
        zdy = (h - zh) // 2
        
        if self.zoom_factor > 1.0:
            max_pan_x = max(0, (zw - w) // 2 + 50)
            max_pan_y = max(0, (zh - h) // 2 + 50)
            px = max(-max_pan_x, min(max_pan_x, self.pan_offset.x()))
            py = max(-max_pan_y, min(max_pan_y, self.pan_offset.y()))
        else:
            px, py = 0, 0
            
        final_x = zdx + px
        final_y = zdy + py
        
        x_img = int((point.x() - final_x) * img_w / zw)
        y_img = int((point.y() - final_y) * img_h / zh)
        
        x_img = max(0, min(img_w - 1, x_img))
        y_img = max(0, min(img_h - 1, y_img))
        return QPoint(x_img, y_img)

    def image_to_widget_coords(self, point: QPoint) -> QPoint:
        if not self.img_before:
            return QPoint(0, 0)
        w = self.width()
        h = self.height()
        img_w = self.img_before.width()
        img_h = self.img_before.height()
        scale = min(w / img_w, h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        zw = int(draw_w * self.zoom_factor)
        zh = int(draw_h * self.zoom_factor)
        zdx = (w - zw) // 2
        zdy = (h - zh) // 2
        
        if self.zoom_factor > 1.0:
            max_pan_x = max(0, (zw - w) // 2 + 50)
            max_pan_y = max(0, (zh - h) // 2 + 50)
            px = max(-max_pan_x, min(max_pan_x, self.pan_offset.x()))
            py = max(-max_pan_y, min(max_pan_y, self.pan_offset.y()))
        else:
            px, py = 0, 0
            
        final_x = zdx + px
        final_y = zdy + py
        
        x_widget = int(final_x + point.x() * zw / img_w)
        y_widget = int(final_y + point.y() * zh / img_h)
        return QPoint(x_widget, y_widget)

    def wheelEvent(self, event):
        if not self.img_before:
            return
        angle = event.angleDelta().y()
        old_zoom = self.zoom_factor
        if angle > 0:
            self.zoom_factor = min(10.0, self.zoom_factor * 1.15)
        else:
            self.zoom_factor = max(1.0, self.zoom_factor / 1.15)
        
        if self.zoom_factor == 1.0:
            self.pan_offset = QPoint(0, 0)
        else:
            # Zoom centered on mouse cursor
            w = self.width()
            h = self.height()
            widget_center = QPoint(w // 2, h // 2)
            mouse_pos = event.position().toPoint()
            delta_mouse = mouse_pos - widget_center
            self.pan_offset = delta_mouse - (delta_mouse - self.pan_offset) * (self.zoom_factor / old_zoom)
        self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.img_before:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.zoom_factor > 1.0:
                self.zoom_factor = 1.0
                self.pan_offset = QPoint(0, 0)
            else:
                old_zoom = 1.0
                self.zoom_factor = 3.0
                w = self.width()
                h = self.height()
                widget_center = QPoint(w // 2, h // 2)
                mouse_pos = event.position().toPoint()
                delta_mouse = mouse_pos - widget_center
                self.pan_offset = delta_mouse - (delta_mouse - self.pan_offset) * (self.zoom_factor / old_zoom)
            self.update()

    def mousePressEvent(self, event):
        if not self.img_before:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            w = self.width()
            line_x = self.split_ratio * w
            if self.show_split and abs(event.position().x() - line_x) < 20:
                self.dragging_split = True
            else:
                p_img = self.widget_to_image_coords(event.position().toPoint())
                if self.roi_rect and self.roi_rect.contains(p_img):
                    self.dragging_roi = True
                    self.roi_drag_start_rect = QRect(self.roi_rect)
                    self.roi_drag_start_img_pos = p_img
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.dragging_pan = True
                    self.last_mouse_pos = event.position()
                    if self.zoom_factor > 1.0:
                        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_selecting_roi = True
            self.roi_start = event.position().toPoint()
            self.roi_end = event.position().toPoint()
            self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseMoveEvent(self, event):
        if not self.img_before:
            return
        w = self.width()
        line_x = self.split_ratio * w
        
        if self.dragging_split:
            self.split_ratio = max(0.0, min(1.0, event.position().x() / w))
            self.update()
        elif self.dragging_roi:
            p_img = self.widget_to_image_coords(event.position().toPoint())
            delta = p_img - self.roi_drag_start_img_pos
            
            img_w = self.img_before.width()
            img_h = self.img_before.height()
            
            new_x = self.roi_drag_start_rect.x() + delta.x()
            new_y = self.roi_drag_start_rect.y() + delta.y()
            
            new_x = max(0, min(img_w - self.roi_drag_start_rect.width(), new_x))
            new_y = max(0, min(img_h - self.roi_drag_start_rect.height(), new_y))
            
            self.roi_rect.moveTo(new_x, new_y)
            self.update()
            if self.roi_changed_cb:
                self.roi_changed_cb()
        elif self.dragging_pan:
            delta = event.position() - self.last_mouse_pos
            self.pan_offset += QPoint(int(delta.x()), int(delta.y()))
            self.last_mouse_pos = event.position()
            self.update()
        elif self.is_selecting_roi:
            self.roi_end = event.position().toPoint()
            self.update()
        else:
            if abs(event.position().x() - line_x) < 15:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                p_img = self.widget_to_image_coords(event.position().toPoint())
                if self.roi_rect and self.roi_rect.contains(p_img):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                elif self.zoom_factor > 1.0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging_split = False
            self.dragging_pan = False
            self.dragging_roi = False
            self.roi_drag_start_rect = None
            self.roi_drag_start_img_pos = None
            
            w = self.width()
            line_x = self.split_ratio * w
            if abs(event.position().x() - line_x) < 15:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                p_img = self.widget_to_image_coords(event.position().toPoint())
                if self.roi_rect and self.roi_rect.contains(p_img):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                elif self.zoom_factor > 1.0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.RightButton and self.is_selecting_roi:
            self.is_selecting_roi = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            p1 = self.widget_to_image_coords(self.roi_start)
            p2 = self.widget_to_image_coords(self.roi_end)
            
            x = min(p1.x(), p2.x())
            y = min(p1.y(), p2.y())
            rw = abs(p1.x() - p2.x())
            rh = abs(p1.y() - p2.y())
            
            if rw > 10 and rh > 10:
                self.roi_rect = QRect(x, y, rw, rh)
            else:
                self.roi_rect = None
            
            self.update()
            if self.roi_changed_cb:
                self.roi_changed_cb()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        if not self.img_before:
            painter.fillRect(self.rect(), QColor("#1e1e1e"))
            painter.setPen(QPen(QColor("#8e8e93")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Preview Image")
            return

        img_w = self.img_before.width()
        img_h = self.img_before.height()
        scale = min(w / img_w, h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)

        # Apply Zoom
        zw = int(draw_w * self.zoom_factor)
        zh = int(draw_h * self.zoom_factor)

        # Center draw coordinates
        zdx = (w - zw) // 2
        zdy = (h - zh) // 2

        # Clamp panning offset to prevent image going entirely off-screen
        if self.zoom_factor > 1.0:
            max_pan_x = max(0, (zw - w) // 2 + 50)
            max_pan_y = max(0, (zh - h) // 2 + 50)
            px = max(-max_pan_x, min(max_pan_x, self.pan_offset.x()))
            py = max(-max_pan_y, min(max_pan_y, self.pan_offset.y()))
            self.pan_offset.setX(px)
            self.pan_offset.setY(py)
        else:
            self.pan_offset = QPoint(0, 0)

        final_x = zdx + self.pan_offset.x()
        final_y = zdy + self.pan_offset.y()

        # Fill background
        painter.fillRect(self.rect(), QColor("#121212"))

        # Enable smooth transformation for drawing to keep it sharp and fast
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # If after image is not loaded, just draw before image zoomed/panned
        if not self.img_after:
            painter.drawImage(QRect(final_x, final_y, zw, zh), self.img_before)
        elif not self.show_split:
            # Just draw after image zoomed/panned across the entire widget
            painter.drawImage(QRect(final_x, final_y, zw, zh), self.img_after)
        else:
            # Physical split x on widget
            split_x = int(self.split_ratio * w)

            # 1. Draw before image (Left side clipped)
            painter.save()
            painter.setClipRect(QRect(0, 0, split_x, h))
            painter.drawImage(QRect(final_x, final_y, zw, zh), self.img_before)
            painter.restore()

            # 2. Draw after image (Right side clipped)
            painter.save()
            painter.setClipRect(QRect(split_x, 0, w, h))
            painter.drawImage(QRect(final_x, final_y, zw, zh), self.img_after)
            painter.restore()

            # 3. Draw vertical split line across the whole widget height
            pen = QPen(QColor("#ffffff"), 2)
            painter.setPen(pen)
            painter.drawLine(split_x, 0, split_x, h)

            # Draw circle handle in the middle of the line
            circle_r = 14
            cy = h // 2
            painter.setBrush(QBrush(QColor("#1c1c1e")))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(split_x - circle_r, cy - circle_r, circle_r * 2, circle_r * 2)

            # Draw simple arrows inside handle
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawLine(split_x - 5, cy, split_x - 2, cy - 3)
            painter.drawLine(split_x - 5, cy, split_x - 2, cy + 3)
            painter.drawLine(split_x + 5, cy, split_x + 2, cy - 3)
            painter.drawLine(split_x + 5, cy, split_x + 2, cy + 3)

        # Draw ROI selection rectangle
        if self.is_selecting_roi and self.roi_start and self.roi_end:
            painter.save()
            painter.setPen(QPen(QColor(10, 132, 255, 180), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = QRect(self.roi_start, self.roi_end)
            painter.drawRect(r)
            painter.restore()
        elif self.roi_rect:
            painter.save()
            p1 = self.image_to_widget_coords(self.roi_rect.topLeft())
            p2 = self.image_to_widget_coords(self.roi_rect.bottomRight())
            r = QRect(p1, p2)
            painter.setPen(QPen(QColor(10, 132, 255, 180), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)
            
            painter.setPen(QPen(QColor("#aeaeb2")))
            painter.drawText(r.topLeft() + QPoint(4, -4), "Preview Area")
            painter.restore()

        # Draw zoom text overlay
        if self.zoom_factor > 1.0:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            zoom_text = f"Zoom: {int(self.zoom_factor * 100)}%"
            font = QFont("SF Pro Display", 10, QFont.Weight.Bold)
            painter.setFont(font)
            
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(zoom_text)
            text_height = fm.height()
            
            padding_x = 8
            padding_y = 4
            bubble_w = text_width + padding_x * 2
            bubble_h = text_height + padding_y * 2
            
            bubble_x = w - bubble_w - 12
            bubble_y = 12
            
            path = QPainterPath()
            path.addRoundedRect(bubble_x, bubble_y, bubble_w, bubble_h, 6, 6)
            
            painter.fillPath(path, QColor(0, 0, 0, 180))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawPath(path)
            
            painter.drawText(bubble_x + padding_x, bubble_y + padding_y + fm.ascent(), zoom_text)
            painter.restore()

# ============================================================================
# Main GUI
# ============================================================================

_DARK_STYLESHEET = """
    QMainWindow, QWidget {
        color: #EDEDEF;
        font-family: 'SF Pro Text', 'Inter', 'Segoe UI', sans-serif;
        font-size: 9.5pt;
    }
    QMainWindow { background: #151516; }
    QWidget#centralSurface { background: #151516; }
    /* Layout helper widgets must not paint their own background.  Only
       explicit cards, panels and controls below receive a surface colour. */
    QWidget { background: transparent; }
    QWidget#controlPanel { background: transparent; }
    QLabel {
        color: #A7A7AC;
        background: transparent;
    }
    QWidget#heroCard {
        background: #1D1D1F;
        border: 1px solid #323236;
        border-radius: 8px;
    }
    QFrame#previewCard {
        background: #101011;
        border: 1px solid #303034;
        border-radius: 8px;
    }
    QFrame#viewportToolbar {
        background: #1D1D1F;
        border: none;
        border-bottom: 1px solid #303034;
        border-radius: 0px;
    }
    QWidget#modelRow {
        background: #222224;
        border: 1px solid #303034;
        border-radius: 6px;
    }
    QLabel#eyebrow {
        color: #85858B;
        font-size: 8pt;
        font-weight: 600;
        letter-spacing: 0.8px;
    }
    QLabel#mutedLabel { color: #77777D; font-size: 9pt; }
    QWidget#statusCard {
        background-color: #1D1D1F;
        border: 1px solid #303034;
        border-radius: 8px;
    }
    QGroupBox {
        background-color: #1D1D1F;
        border: 1px solid #303034;
        border-radius: 7px;
        margin-top: 10px;
        padding: 14px 10px 10px 10px;
        font-weight: bold;
        color: #EDEDEF;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 14px;
        padding: 0px 8px;
        color: #D6D6D9;
        background-color: #1D1D1F;
    }
    QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #252527;
        border: 1px solid #3A3A3E;
        border-radius: 5px;
        padding: 4px 8px;
        color: #EDEDEF;
        min-height: 22px;
    }
    QSpinBox, QDoubleSpinBox {
        padding-right: 22px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #3A3A3E;
        border-top-right-radius: 5px;
        background-color: #303033;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid #3A3A3E;
        border-bottom-right-radius: 5px;
        background-color: #303033;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #444448;
    }
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
        border-color: #5E9EFF;
    }
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border-color: #5E9EFF;
    }
    QCheckBox {
        spacing: 8px;
        color: #D6D6D9;
    }
    QCheckBox:hover {
        color: #FFFFFF;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid #4A4A4F;
        background-color: #252527;
    }
    QCheckBox::indicator:hover {
        border-color: #5E9EFF;
    }
    QCheckBox::indicator:checked {
        background-color: #3478D4;
        border-color: #5E9EFF;
    }
    QSlider::groove:horizontal {
        background: #343438;
        height: 4px;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #FFFFFF;
        border: 2px solid #5E9EFF;
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        border-color: #8BB9FF;
        background: #8BB9FF;
    }
    QSlider::sub-page:horizontal {
        background: #5E9EFF;
        border-radius: 2px;
    }
    QProgressBar {
        background-color: #252527;
        border: 1px solid #3A3A3E;
        border-radius: 5px;
        text-align: center;
        color: #FFFFFF;
        font-weight: bold;
        font-size: 8pt;
        min-height: 16px;
    }
    QProgressBar::chunk {
        background: #5E9EFF;
        border-radius: 5px;
    }
    QPushButton {
        background-color: #2A2A2D;
        border: 1px solid #3B3B40;
        border-radius: 5px;
        padding: 5px 13px;
        color: #E6E6E9;
        font-weight: 500;
        min-height: 22px;
    }
    QPushButton:hover {
        background-color: #353539;
        border-color: #55555B;
        color: #FFFFFF;
    }
    QPushButton:pressed {
        background-color: #404044;
        color: #FFFFFF;
    }
    QPushButton:disabled {
        background-color: #202022;
        color: #66666C;
        border-color: #2A2A2D;
    }
    QPushButton#primaryBtn {
        background: #3478D4;
        color: #FFFFFF;
        font-weight: bold;
        font-size: 10pt;
        padding: 8px 22px;
        border: none;
        border-radius: 5px;
    }
    QPushButton#primaryBtn:hover {
        background: #4388E6;
    }
    QPushButton#primaryBtn:pressed {
        background-color: #2868BE;
    }
    QPushButton#primaryBtn:disabled {
        background: #272729;
        color: #66666C;
    }
    QPushButton#successBtn {
        background: #287A53;
        border: 1px solid #359768;
        color: #FFFFFF;
        font-weight: 600;
    }
    QPushButton#successBtn:hover { background: #309061; }
    QPushButton#successBtn:disabled {
        background: #242426;
        border-color: #303034;
        color: #66666C;
    }
    QPushButton#cancelBtn {
        background-color: transparent;
        border: 1px solid #6A3B42;
        color: #E97883;
    }
    QPushButton#cancelBtn:hover {
        background-color: #A83D49;
        color: #FFFFFF;
    }
    QScrollBar:vertical {
        border: none;
        background: #1B1B1D;
        width: 8px;
        margin: 0px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #46464B;
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #66666C;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: none;
        background: none;
        height: 0px;
    }
    QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
        border: none;
        background: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }
    QTabWidget::pane {
        border: 1px solid #303034;
        background-color: #1D1D1F;
        border-radius: 7px;
        top: -1px;
        padding: 6px;
    }
    QTabWidget::tab-bar {
        alignment: left;
    }
    QTabBar::tab {
        background-color: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 14px;
        font-weight: 500;
        font-size: 11px;
        color: #8D8D93;
        margin-right: 4px;
        min-width: 80px;
        text-align: center;
    }
    QTabBar::tab:hover {
        background-color: #242426;
        color: #D6D6D9;
    }
    QTabBar::tab:selected {
        background-color: transparent;
        border-bottom-color: #5E9EFF;
        color: #FFFFFF;
    }
    QToolTip {
        background: #111827;
        color: #F8FAFC;
        border: 1px solid #334155;
        padding: 6px 8px;
    }
"""


class AnimatedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(250)
        self.anim.valueChanged.connect(self._update_val)

    def _update_val(self, val):
        super().setValue(val)

    def setValue(self, val):
        self.anim.stop()
        self.anim.setStartValue(self.value())
        self.anim.setEndValue(val)
        self.anim.start()


class DiagnosticDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QPlainTextEdit
        
        self.setWindowTitle("SyQon Parallax - System & GPU Diagnostic")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1c;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPlainTextEdit {
                background-color: #121214;
                color: #30d158;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #3a3a3c;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #2c2c2e;
                color: #ffffff;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #3a3a3c;
            }
            QPushButton:hover {
                background-color: #3a3a3c;
            }
            QPushButton#btn_copy {
                background-color: #0a84ff;
                border: none;
            }
            QPushButton#btn_copy:hover {
                background-color: #007aff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        self.lbl_title = QLabel("<b>Hardware & AI Engine Diagnostics</b>")
        self.lbl_title.setStyleSheet("font-size: 14px; color: #0a84ff;")
        layout.addWidget(self.lbl_title)

        self.lbl_desc = QLabel(
            "This utility tests your GPU and PyTorch compatibility. "
            "Please copy the logs below and send them to support to resolve issues."
        )
        self.lbl_desc.setWordWrap(True)
        layout.addWidget(self.lbl_desc)

        self.log_area = QPlainTextEdit(self)
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Tests", self)
        self.btn_run.clicked.connect(self.run_diagnostics)
        
        self.btn_copy = QPushButton("Copy Logs to Clipboard", self)
        self.btn_copy.setObjectName("btn_copy")
        self.btn_copy.clicked.connect(self.copy_logs)
        self.btn_copy.setEnabled(False)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_copy)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def run_diagnostics(self):
        self.btn_run.setEnabled(False)
        self.log_area.setPlainText("Running diagnostics, please wait...\n")
        QApplication.processEvents()

        logs = []
        logs.append("=========================================")
        logs.append("       SYQON PARALLAX DIAGNOSTIC LOG     ")
        logs.append("=========================================\n")

        # 1. System Info
        logs.append("[SYSTEM INFO]")
        logs.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
        logs.append(f"Python Version: {sys.version.split()[0]}")
        try:
            import PySide6
            logs.append(f"PySide6 Version: {PySide6.__version__}")
        except Exception:
            logs.append("PySide6 Version: Unknown")
        logs.append("")

        # 2. PyTorch Info
        logs.append("[PYTORCH INFO]")
        try:
            version_str = torch.__version__
            base_version = version_str.split('+')[0].split('a')[0].split('b')[0].split('rc')[0]
            try:
                parts = [int(p) for p in base_version.split('.') if p.isdigit()]
                if len(parts) >= 2:
                    major, minor = parts[0], parts[1]
                    if major > 2 or (major == 2 and minor >= 0):
                        compat = "COMPATIBLE (Modern version - PyTorch 2.x+ full support)"
                    elif major == 1 and minor >= 8:
                        compat = "COMPATIBLE (Older version - limited performance, upgrade to 2.x+ recommended)"
                    else:
                        compat = "INCOMPATIBLE (Too old - PyTorch 1.8+ required, please upgrade)"
                else:
                    compat = "COMPATIBLE (Assuming compatible)"
            except Exception:
                compat = "COMPATIBLE (Version check skipped)"
            logs.append(f"PyTorch Version: {version_str} -> {compat}")
            hip_version = getattr(torch.version, "hip", None)
            native_gpu_api = "ROCm/HIP" if hip_version else "CUDA"
            logs.append(f"{native_gpu_api} Available: {torch.cuda.is_available()}")
            if hip_version:
                logs.append(f"ROCm/HIP Version: {hip_version}")
            if torch.cuda.is_available():
                logs.append(f"{native_gpu_api} Device Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logs.append(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                    try:
                        cap = torch.cuda.get_device_capability(i)
                        logs.append(f"    Capability: {cap[0]}.{cap[1]}")
                    except Exception:
                        pass
            
            # DirectML Info
            dml_available = False
            try:
                import torch_directml
                dml_available = torch_directml.is_available()
                logs.append(f"DirectML Available: {dml_available}")
            except ImportError:
                logs.append("DirectML Available: False (torch_directml not installed)")
            
            # MPS Info
            mps_available = False
            try:
                mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
                logs.append(f"MPS Available (Metal): {mps_available}")
            except Exception:
                logs.append("MPS Available (Metal): False")

            # Intel XPU Info
            xpu_available = False
            try:
                xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
                logs.append(f"Intel XPU Available: {xpu_available}")
            except Exception:
                logs.append("Intel XPU Available: False")
        except Exception as exc:
            logs.append(f"Error loading PyTorch: {exc}")
        logs.append("")

        # 3. Default Device Selection
        logs.append("[DEFAULT ENGINE DEVICE]")
        try:
            detected_device = pick_device(use_gpu=True)
            logs.append(f"Detected Device: {detected_device}")
        except Exception as exc:
            logs.append(f"Error calling pick_device(): {exc}")
        logs.append("")

        # 4. Hardware Inference & Compatibility Tests
        logs.append("[HARDWARE COMPATIBILITY TESTS]")
        
        devices_to_test = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices_to_test.append(torch.device("cuda"))
        if dml_available:
            try:
                devices_to_test.append(torch_directml.device())
            except Exception:
                pass
        if mps_available:
            devices_to_test.append(torch.device("mps"))
        if xpu_available:
            devices_to_test.append(torch.device("xpu"))

        for dev in devices_to_test:
            logs.append(f"Testing Device: {dev}")
            
            # Test A: Basic tensor allocation and copy
            try:
                t = torch.zeros((1, 3, 64, 64), dtype=torch.float32, device=dev)
                logs.append("  Test A (Allocation): PASS")
            except Exception as exc:
                logs.append(f"  Test A (Allocation): FAIL ({exc})")
                continue
            
            # Test B: Convolution operation (Conv2d)
            try:
                conv = nn.Conv2d(3, 8, kernel_size=3, padding=1).to(dev)
                with torch.no_grad():
                    out = conv(t)
                logs.append(f"  Test B (Conv2D): PASS (Output shape: {list(out.shape)})")
                del conv, out
            except Exception as exc:
                logs.append(f"  Test B (Conv2D): FAIL ({exc})")
                
            # Test C: LayerNorm + Permute Contiguity Test (DirectML bug verification)
            try:
                ln = nn.LayerNorm(8).to(dev)
                t_test = torch.randn(1, 8, 16, 16, device=dev)
                t_perm = t_test.permute(0, 2, 3, 1) # Non-contiguous
                try:
                    out_ln = ln(t_perm)
                    is_nan = torch.isnan(out_ln).any().item()
                    if is_nan:
                        logs.append("  Test C (LayerNorm strided): FAIL (Output contains NaN)")
                    else:
                        logs.append("  Test C (LayerNorm strided): PASS")
                except Exception as inner_exc:
                    logs.append(f"  Test C (LayerNorm strided): FAIL ({inner_exc})")
                del ln, t_test, t_perm
            except Exception as exc:
                logs.append(f"  Test C (LayerNorm strided): FAIL ({exc})")
                
            # Test D: LayerNorm with contiguous fix
            try:
                ln = nn.LayerNorm(8).to(dev)
                t_test = torch.randn(1, 8, 16, 16, device=dev)
                t_perm = t_test.permute(0, 2, 3, 1).contiguous()
                out_ln = ln(t_perm)
                is_nan = torch.isnan(out_ln).any().item()
                if is_nan:
                    logs.append("  Test D (LayerNorm contiguous): FAIL (Output contains NaN)")
                else:
                    logs.append("  Test D (LayerNorm contiguous): PASS")
                del ln, t_test, t_perm, out_ln
            except Exception as exc:
                logs.append(f"  Test D (LayerNorm contiguous): FAIL ({exc})")

            # Clean memory
            del t
            _empty_device_cache(dev)

            logs.append("")

        logs.append("=========================================")
        logs.append("            DIAGNOSTICS COMPLETED        ")
        logs.append("=========================================")

        full_log = "\n".join(logs)
        self.log_area.setPlainText(full_log)
        self.btn_run.setEnabled(True)
        self.btn_copy.setEnabled(True)

    def copy_logs(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_area.toPlainText())
        self.btn_copy.setText("Logs Copied!")
        QTimer.singleShot(2000, lambda: self.btn_copy.setText("Copy Logs to Clipboard"))


class ParallaxGUI(QMainWindow):
    def __init__(self, siril, config: dict, raw: np.ndarray,
                 orig_dtype, scale: float, orig_was_mono: bool,
                 engine_dir: Path, flip_vertical: bool = False):
        super().__init__()
        self.siril          = siril
        self.config         = dict(config)
        self.raw            = raw
        self.orig_dtype     = orig_dtype
        self.scale          = scale
        self.orig_was_mono  = orig_was_mono
        self.flip_vertical  = flip_vertical

        # Siril's embedded Qt plugin does not support window-opacity changes.
        # Show the window immediately instead of using a fade-in animation.
        self.engine_dir     = engine_dir
        self._worker        = None
        self._worker_thread = None
        
        self._update_window_title()

        # Prepare original raw image in RGB float32 for the preview (with correct FITS orientation)
        self.preview_raw, _, _, _ = _prepare_for_inference(self.raw, flip_vertical=self.flip_vertical)
        self.processed_hwc        = None
        self.processed_roi_hwc    = None
        self.processed_roi_rect   = None
        self.pending_siril_result = None

        self._build_ui()
        self._load_config_into_ui()

    def _update_window_title(self):
        edition_label = "Nano" if _CURRENT_EDITION == "nano" else "Pro"
        self.setWindowTitle(f"SyQon Parallax — {edition_label} Edition v{SCRIPT_VERSION}")

    def _update_edition_buttons_style(self):
        active_style = """
            QPushButton {
                background-color: #3478D4;
                color: #ffffff;
                font-weight: 600;
                border: 1px solid #4388E6;
                padding: 2px;
            }
        """
        inactive_style = """
            QPushButton {
                background-color: #252527;
                color: #A7A7AC;
                font-weight: normal;
                border: 1px solid #3A3A3E;
                padding: 2px;
            }
        """
        if _CURRENT_EDITION == "nano":
            self.btn_nano.setStyleSheet(active_style + "QPushButton { border-top-left-radius: 6px; border-bottom-left-radius: 6px; border-top-right-radius: 0px; border-bottom-right-radius: 0px; }")
            self.btn_pro.setStyleSheet(inactive_style + "QPushButton { border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-top-left-radius: 0px; border-bottom-left-radius: 0px; border-left: none; }")
        else:
            self.btn_nano.setStyleSheet(inactive_style + "QPushButton { border-top-left-radius: 6px; border-bottom-left-radius: 6px; border-top-right-radius: 0px; border-bottom-right-radius: 0px; }")
            self.btn_pro.setStyleSheet(active_style + "QPushButton { border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-top-left-radius: 0px; border-bottom-left-radius: 0px; border-left: none; }")

    def _update_star_level_range(self):
        max_level = 7.0 if _CURRENT_EDITION == "pro" else 5.0
        self.lbl_star_title.setText(f"Sharpen Star level (0 = off, 0.1–{int(max_level)}):")
        
        self.lbl_star_subtitle.setText("")
        self.lbl_star_subtitle.hide()

        if _CURRENT_EDITION == "pro":
            self.btn_get.setText("Get Models Here…")
            self.lbl_batch.show()
            self.combo_batch.show()
        else:
            self.btn_get.setText("Get Free Models")
            self.lbl_batch.hide()
            self.combo_batch.hide()

        self.spin_star.blockSignals(True)
        self.sld_star.blockSignals(True)
        
        self.spin_star.setRange(0.0, max_level)
        self.sld_star.setRange(0, int(max_level * 10))
        
        # Clamp current value to new max if needed
        val = min(self.spin_star.value(), max_level)
        self.spin_star.setValue(val)
        self.sld_star.setValue(int(round(val * 10.0)))
        
        self.spin_star.blockSignals(False)
        self.sld_star.blockSignals(False)

    def _update_sharpen_range(self):
        self.sld_sharpen.blockSignals(True)
        real_sharpen = float(self.config.get("sharpen_alpha", 1.0))
        
        if _CURRENT_EDITION == "pro":
            self.sld_sharpen.setRange(0, 300)
            slider_val = int(round(real_sharpen * 100.0))
            slider_val = min(max(slider_val, 0), 300)
        else:
            self.sld_sharpen.setRange(0, 300)
            # map real_sharpen (0.0-2.25) to slider (0.0-3.0) where 2.0 UI maps to 1.5 real
            slider_val = int(round((real_sharpen * (2.0 / 1.5)) * 100.0))
            slider_val = min(max(slider_val, 0), 300)
            
        self.sld_sharpen.setValue(slider_val)
        self.sld_sharpen.blockSignals(False)
        self._on_sharpen_changed(slider_val)

    def _set_edition(self, edition: str):
        global _CURRENT_EDITION
        if _CURRENT_EDITION == edition:
            return
        
        # Save current sharpen_alpha state before switching edition to map it correctly
        cfg = self._gather_config()
        self.config["sharpen_alpha"] = cfg["sharpen_alpha"]
        
        _CURRENT_EDITION = edition
        self.config["edition"] = edition
        save_config(self.config, self.siril)
        
        clear_model_cache()
        self._update_window_title()
        self._update_edition_buttons_style()
        self._update_mode_visibility()
        self._update_star_level_range()
        self._update_sharpen_range()
        
        edition_label = "Nano" if edition == "nano" else "Pro"
        self.lbl_hero_title.setText(f"SyQon Parallax ({edition_label})")
        self._refresh_model_status()
        self._update_preview()
        self.lbl_status.setText(f"Switched to Parallax {edition_label} Edition.")

    def _on_mode_changed(self):
        self.chk_mtf.setChecked(True)
        self.spin_mtf.setValue(0.25)
        self.chk_linked.setChecked(False)
        self.config = self._gather_config()
        save_config(self.config, self.siril)
        clear_model_cache()
        self._refresh_model_status()
        self._update_star_level_range()
        self._update_preview()

    def _update_mode_visibility(self):
        if _CURRENT_EDITION == "pro":
            self.mode_widget.show()
            self.lbl_classic_section.show()
            self.lbl_aesthetics_section.show()
            # Pro exposes two complete, independent three-model sets.
            # Never hide the Classic star-reduction checkpoint: doing so made
            # the installer appear to support only five Pro models.
            for _key, w in self.classic_widgets:
                w.show()
            for w in self.aesthetics_widgets:
                w.show()
        else:
            self.mode_widget.hide()
            self.combo_mode.blockSignals(True)
            self.combo_mode.setCurrentIndex(0)
            self.combo_mode.blockSignals(False)
            self.lbl_classic_section.hide()
            self.lbl_aesthetics_section.hide()
            for key, w in self.classic_widgets:
                w.show()
            for w in self.aesthetics_widgets:
                w.hide()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.setStyleSheet(_DARK_STYLESHEET)
        # Force an opaque backing surface. Some Linux/Siril embedded Qt
        # backends propagate transparent helper widgets to the whole window.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        # Keep the complete workspace usable on laptop and small desktop
        # displays. The split panels and scroll areas handle tighter layouts.
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        target_w = min(1280, max(720, (available.width() - 24) if available else 1280))
        target_h = min(820, max(520, (available.height() - 48) if available else 820))
        self.setMinimumSize(720, 520)
        self.resize(target_w, target_h)
        central = QWidget()
        central.setObjectName("centralSurface")
        central.setAutoFillBackground(True)
        self.setCentralWidget(central)
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(12, 12, 12, 10)
        main_lay.setSpacing(10)
        
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(6)
        main_lay.addWidget(self.workspace_splitter, 1)

        # LEFT CONTROL PANEL (Width: 450px)
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setMinimumWidth(290)
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFrameShape(QFrame.NoFrame)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        ctrl_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        ctrl_panel = QWidget()
        ctrl_panel.setObjectName("controlPanel")
        ctrl_lay = QVBoxLayout(ctrl_panel)
        ctrl_lay.setContentsMargins(0, 0, 0, 0)
        ctrl_lay.setSpacing(4)

        # Header
        hero = QFrame(); hero.setObjectName("heroCard")
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(13, 10, 13, 10)
        
        header_row = QHBoxLayout()
        self.lbl_hero_title = QLabel("SyQon Parallax")
        self.lbl_hero_title.setStyleSheet("color:#F2F2F4; font-size:14pt; font-weight:600;")
        
        self.edition_layout = QHBoxLayout()
        self.edition_layout.setSpacing(0)
        
        self.btn_nano = QPushButton("Nano")
        self.btn_nano.setFixedWidth(60)
        self.btn_nano.setFixedHeight(25)
        self.btn_nano.clicked.connect(lambda: self._set_edition("nano"))
        
        self.btn_pro = QPushButton("Pro")
        self.btn_pro.setFixedWidth(60)
        self.btn_pro.setFixedHeight(25)
        self.btn_pro.clicked.connect(lambda: self._set_edition("pro"))
        
        self.edition_layout.addWidget(self.btn_nano)
        self.edition_layout.addWidget(self.btn_pro)
        
        header_row.addWidget(self.lbl_hero_title)
        header_row.addStretch()
        header_row.addLayout(self.edition_layout)
        
        sub = QLabel("Neural image processing for Siril")
        sub.setStyleSheet("color:#85858B; font-size:9pt;")
        
        hero_lay.addLayout(header_row)
        hero_lay.addWidget(sub)
        ctrl_lay.addWidget(hero)

        # Processing Pipeline
        g_pipeline = QGroupBox("Neural Processing Pipeline")
        lay_pipe = QVBoxLayout(g_pipeline)
        lay_pipe.setContentsMargins(10, 10, 10, 8)
        lay_pipe.setSpacing(4)

        # Mode Selection widget (Visible only in Pro edition)
        self.mode_widget = QWidget()
        mode_lay = QHBoxLayout(self.mode_widget)
        mode_lay.setContentsMargins(0, 0, 0, 4)
        lbl_mode = QLabel("Pipeline Mode:")
        lbl_mode.setStyleSheet("color:#ffffff; font-weight:bold; font-size:10pt;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Classic (Natural)", "Defined"])
        self.combo_mode.setStyleSheet("QComboBox { font-weight:600; }")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_lay.addWidget(lbl_mode)
        mode_lay.addWidget(self.combo_mode)
        mode_lay.addStretch()
        lay_pipe.addWidget(self.mode_widget)

        # Aberration Correction
        self.chk_correct = QCheckBox("Enable Aberration Correction")
        self.chk_correct.setToolTip("Stage 02 — Correct geometric and chromatic stellar aberrations.")
        lay_pipe.addWidget(self.chk_correct)

        # Sharpen Star
        star_ctrl_lay = QHBoxLayout()
        self.lbl_star_title = QLabel("Sharpen Star level (0 = off, 1–6):")
        self.spin_star = QDoubleSpinBox()
        self.spin_star.setDecimals(1)
        self.spin_star.setSingleStep(0.1)
        self.spin_star.setRange(0.0, 6.0)
        self.spin_star.setFixedWidth(80)
        star_ctrl_lay.addWidget(self.lbl_star_title)
        star_ctrl_lay.addWidget(self.spin_star)
        star_ctrl_lay.addStretch()
        lay_pipe.addLayout(star_ctrl_lay)
        
        star_slider_lay = QHBoxLayout()
        star_slider_lbl = QLabel("Adjust level:")
        self.sld_star = QSlider(Qt.Orientation.Horizontal)
        self.sld_star.setRange(0, 60)
        self.sld_star.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_star.setTickInterval(10)
        star_slider_lay.addWidget(star_slider_lbl)
        star_slider_lay.addWidget(self.sld_star, 1)
        
        # Sync spinbox and slider (Double value)
        self.spin_star.valueChanged.connect(lambda val: (
            self.sld_star.blockSignals(True),
            self.sld_star.setValue(int(round(val * 10.0))),
            self.sld_star.blockSignals(False)
        ))
        self.sld_star.valueChanged.connect(lambda val: (
            self.spin_star.blockSignals(True),
            self.spin_star.setValue(val / 10.0),
            self.spin_star.blockSignals(False)
        ))
        lay_pipe.addLayout(star_slider_lay)

        # Star Subtitle (Pro-only note)
        self.lbl_star_subtitle = QLabel("")
        self.lbl_star_subtitle.setStyleSheet("font-size: 11px; color: #8e8e93; font-style: italic; margin-top: -2px; margin-bottom: 4px;")
        lay_pipe.addWidget(self.lbl_star_subtitle)

        # Sharpening
        self.chk_sharpen = QCheckBox("Enable Nebula Deblur")
        self.chk_sharpen.setToolTip("Stage 01 — Restore fine structures in nebulae and extended objects.")
        lay_pipe.addWidget(self.chk_sharpen)

        sh_row = QHBoxLayout()
        self.lbl_sharpen_strength = QLabel("Nebula deblur strength:")
        sh_row.addWidget(self.lbl_sharpen_strength)
        self.sld_sharpen = QSlider(Qt.Orientation.Horizontal)
        self.sld_sharpen.setRange(0, 300)
        self.lbl_sharpen = QLabel("1.00")
        self.lbl_sharpen.setFixedWidth(40)
        self.lbl_sharpen.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sld_sharpen.valueChanged.connect(self._on_sharpen_changed)
        sh_row.addWidget(self.sld_sharpen, 1)
        sh_row.addWidget(self.lbl_sharpen)
        lay_pipe.addLayout(sh_row)

        self.lbl_sharpen_warn = QLabel("")
        self.lbl_sharpen_warn.setStyleSheet("color:#ff453a; font-size:8pt; font-style:italic;")
        self.lbl_sharpen_warn.setWordWrap(True)
        lay_pipe.addWidget(self.lbl_sharpen_warn)
        
        self.chk_sharpen.toggled.connect(lambda on: (
            self.sld_sharpen.setEnabled(on),
            self.lbl_sharpen.setEnabled(on),
            self.lbl_sharpen_strength.setEnabled(on)
        ))
        
        # Live tile update toggle
        self.chk_live_update = QCheckBox("Enable real-time preview updates")
        self.chk_live_update.setStyleSheet("QCheckBox { margin-top: 10px; font-weight: bold; }")
        lay_pipe.addWidget(self.chk_live_update)
        
        self.lbl_live_update_desc = QLabel(
            "Disabling real-time updates can improve processing speed, "
            "especially on slower hardware or when running on CPU/DirectML."
        )
        self.lbl_live_update_desc.setWordWrap(True)
        self.lbl_live_update_desc.setStyleSheet("color: #ffffff; font-size: 10px; font-style: italic; margin-top: 1px; margin-bottom: 4px;")
        lay_pipe.addWidget(self.lbl_live_update_desc)
        
        
        # Advanced Settings
        g_advanced = QGroupBox("Advanced & Performance")
        lay_adv = QVBoxLayout(g_advanced)
        lay_adv.setContentsMargins(10, 10, 10, 8)
        lay_adv.setSpacing(4)

        self.chk_unlock_advanced = QCheckBox("Unlock tiling geometry settings")
        self.chk_unlock_advanced.setStyleSheet("QCheckBox { font-weight: bold; color: #ff9f0a; }")
        lay_adv.addWidget(self.chk_unlock_advanced)
        
        self.lbl_warning = QLabel(
            "Tile size, overlap and edge padding affect memory use, seams and processing speed. "
            "Keep the defaults unless you understand the tiling trade-offs."
        )
        self.lbl_warning.setWordWrap(True)
        self.lbl_warning.setStyleSheet("color: #ff453a; font-size: 10px; font-style: italic; margin-top: 2px; margin-bottom: 6px;")
        self.lbl_warning.setVisible(False)
        lay_adv.addWidget(self.lbl_warning)

        # Tiling
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("Tile size:"))
        self.spin_tile = QSpinBox(); self.spin_tile.setRange(128, 2048); self.spin_tile.setSingleStep(64)
        self.spin_tile.setValue(512)
        self.spin_tile.setFixedWidth(80)
        t_row.addWidget(self.spin_tile)
        t_row.addWidget(QLabel("Overlap:"))
        self.spin_overlap = QSpinBox(); self.spin_overlap.setRange(8, 512); self.spin_overlap.setSingleStep(8)
        self.spin_overlap.setValue(64)
        self.spin_overlap.setFixedWidth(80)
        t_row.addWidget(self.spin_overlap)
        t_row.addStretch()
        lay_adv.addLayout(t_row)

        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Edge pad:"))
        self.spin_pad = QSpinBox(); self.spin_pad.setRange(0, 2048); self.spin_pad.setSingleStep(16)
        self.spin_pad.setValue(96)
        self.spin_pad.setFixedWidth(80)
        pad_row.addWidget(self.spin_pad)
        pad_row.addSpacing(20)
        
        self.lbl_batch = QLabel("Batch size:")
        self.combo_batch = QComboBox()
        self.combo_batch.addItems(["Auto", "1", "2", "3", "4", "5"])
        self.combo_batch.setCurrentIndex(0)
        self.combo_batch.setFixedWidth(80)
        pad_row.addWidget(self.lbl_batch)
        pad_row.addWidget(self.combo_batch)
        pad_row.addStretch()
        lay_adv.addLayout(pad_row)

        # Stretch
        self.chk_mtf = QCheckBox("Apply temporary stretch (for linear data)")
        lay_adv.addWidget(self.chk_mtf)
        
        mtf_row = QHBoxLayout()
        mtf_row.addWidget(QLabel("Target median:"))
        self.spin_mtf = QDoubleSpinBox()
        self.spin_mtf.setRange(0.01, 0.50)
        self.spin_mtf.setSingleStep(0.01)
        self.spin_mtf.setDecimals(3)
        self.spin_mtf.setFixedWidth(80)
        mtf_row.addWidget(self.spin_mtf)
        mtf_row.addStretch()
        lay_adv.addLayout(mtf_row)
        
        self.chk_linked = QCheckBox("Linked stretch (preserves star colors)")
        lay_adv.addWidget(self.chk_linked)
        self.chk_mtf.toggled.connect(lambda on: (self.spin_mtf.setEnabled(on), self.chk_linked.setEnabled(on)))

        output_separator = QFrame()
        output_separator.setFrameShape(QFrame.Shape.HLine)
        output_separator.setStyleSheet("color:#334155; margin:6px 0 4px 0;")
        lay_adv.addWidget(output_separator)

        output_title = QLabel("PROCESSING & SIRIL HANDOFF")
        output_title.setStyleSheet("color:#94a3b8; font-size:8pt; font-weight:700; letter-spacing:0.6px;")
        lay_adv.addWidget(output_title)

        self.lbl_handoff_desc = QLabel(
            "Pipeline: Aberration Correction → Sharpen Star → Nebula Deblur.\n"
            "Apply to Siril updates the image already loaded in memory: no TIFF/FITS export or "
            "file conversion is performed. The original data type, precision and FITS orientation are restored."
        )
        self.lbl_handoff_desc.setWordWrap(True)
        self.lbl_handoff_desc.setStyleSheet("color:#94a3b8; font-size:9px; margin-bottom:4px;")
        lay_adv.addWidget(self.lbl_handoff_desc)

        # Performance Settings
        self.chk_no_gpu = QCheckBox("Force CPU execution (disable GPU)")
        self.chk_no_gpu.stateChanged.connect(lambda _: clear_model_cache())
        lay_adv.addWidget(self.chk_no_gpu)
        
        # Detected GPU display
        try:
            dev_test = pick_device(use_gpu=True)
            if dev_test.type == "cpu":
                gpu_name = "None (CPU fallback)"
            elif dev_test.type == "cuda":
                if getattr(torch.version, "hip", None):
                    gpu_name = f"AMD GPU (ROCm/HIP) - {torch.cuda.get_device_name(0)}"
                else:
                    gpu_name = f"NVIDIA GPU (CUDA) - {torch.cuda.get_device_name(0)}"
            elif dev_test.type == "mps":
                gpu_name = "Apple Silicon / AMD GPU (Metal MPS)"
            elif dev_test.type == "xpu":
                gpu_name = "Intel GPU (XPU)"
            else:
                try:
                    import torch_directml
                    gpu_name = "AMD / Intel GPU (DirectML)"
                except Exception:
                    gpu_name = f"Generic GPU ({dev_test})"
        except Exception:
            gpu_name = "Failed to detect GPU"

        self.lbl_gpu_detected = QLabel(f"<b>Active Hardware Accelerator:</b> {gpu_name}")
        self.lbl_gpu_detected.setStyleSheet("color: #30d158; font-size: 11px; margin-top: 4px; margin-bottom: 2px;")
        lay_adv.addWidget(self.lbl_gpu_detected)
        
        self.btn_diagnostic = QPushButton("Run Hardware & AI Diagnostic...")
        self.btn_diagnostic.setFixedHeight(22)
        self.btn_diagnostic.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background-color: #2c2c2e;
                color: #ffffff;
                border-radius: 4px;
                border: 1px solid #3a3a3c;
                margin-top: 4px;
                margin-bottom: 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a3a3c;
            }
        """)
        self.btn_diagnostic.clicked.connect(self._on_run_diagnostic)
        lay_adv.addWidget(self.btn_diagnostic)
        
        self.chk_unlock_advanced.toggled.connect(self._on_unlock_toggled)
        
        
        # AI Model Weights
        self.mbox_models = QGroupBox("AI Model Weights")
        mform = QVBoxLayout(self.mbox_models)
        mform.setContentsMargins(10, 10, 10, 8)
        mform.setSpacing(4)
        
        lbl_subtitle = QLabel("Download models and click 'Select Model Files...' below to auto-install (supports selecting all files at once).")
        lbl_subtitle.setStyleSheet("font-size: 10px; color: #8e8e93; font-style: italic; margin-bottom: 4px;")
        lbl_subtitle.setWordWrap(True)
        mform.addWidget(lbl_subtitle)
        
        self._model_status = {}
        
        # Classic Section Header
        self.lbl_classic_section = QLabel("Classic Models (Classic pipeline — 3 required)")
        self.lbl_classic_section.setStyleSheet("color: #8e8e93; font-weight: bold; font-size: 10px; margin-top: 4px; margin-bottom: 2px;")
        mform.addWidget(self.lbl_classic_section)
        
        # Classic Models
        self.classic_widgets = []
        for key, label, path_fn in [
            ("classic_correction",   "Correction",     lambda: _correction_path("classic")),
            ("classic_star_reduce",  "Sharpen Star",   lambda: _star_reduce_path("classic")),
            ("classic_sharpen",      "Sharpen",        lambda: _sharpen_path("classic")),
        ]:
            col_widget, slbl = _make_model_row(
                self.mbox_models, label, path_fn(),
                lambda checked=False, k=key: self._remove_model(k),
            )
            self._model_status[key] = slbl
            mform.addWidget(col_widget)
            self.classic_widgets.append((key, col_widget))

        # Aesthetics Section Header
        self.lbl_aesthetics_section = QLabel("Defined Models (Defined pipeline — 3 required)")
        self.lbl_aesthetics_section.setStyleSheet("color: #8e8e93; font-weight: bold; font-size: 10px; margin-top: 8px; margin-bottom: 2px;")
        mform.addWidget(self.lbl_aesthetics_section)

        # Aesthetics Models
        self.aesthetics_widgets = []
        for key, label, path_fn in [
            ("aesthetics_correction",   "Correction (StarOnly)",     lambda: _correction_path("aesthetics")),
            ("aesthetics_star_reduce",  "Sharpen Star",              lambda: _star_reduce_path("aesthetics")),
            ("aesthetics_sharpen",      "Sharpen (Deblur Nebula)",   lambda: _sharpen_path("aesthetics")),
        ]:
            col_widget, slbl = _make_model_row(
                self.mbox_models, label, path_fn(),
                lambda checked=False, k=key: self._remove_model(k),
            )
            self._model_status[key] = slbl
            mform.addWidget(col_widget)
            self.aesthetics_widgets.append(col_widget)

        self.btn_select_models = QPushButton("Select Model Files...")
        self.btn_select_models.clicked.connect(self._on_select_models_clicked)
        self.btn_select_models.setFixedHeight(26)
        self.btn_select_models.setStyleSheet("QPushButton { font-weight: bold; background-color: #007aff; color: white; border-radius: 4px; font-size: 11px; min-height: 20px; } QPushButton:hover { background-color: #0a84ff; }")
        mform.addWidget(self.btn_select_models)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        actions_row.setContentsMargins(0, 0, 0, 0)
        
        self.btn_get = QPushButton("")
        self.btn_get.clicked.connect(self._on_get_models_clicked)
        self.btn_get.setFixedHeight(22)
        self.btn_get.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; min-height: 18px; font-weight: normal; }")
        
        btn_cc  = QPushButton("Clear AI Cache")
        btn_cc.clicked.connect(
            lambda: (clear_model_cache(), self.lbl_status.setText("AI cache cleared."))
        )
        btn_cc.setFixedHeight(22)
        btn_cc.setStyleSheet("QPushButton { font-size: 10px; padding: 2px 6px; min-height: 18px; font-weight: normal; }")
        
        actions_row.addWidget(self.btn_get, 1)
        actions_row.addWidget(btn_cc, 1)
        mform.addLayout(actions_row)
        
        # Create Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setMovable(False)
        
        # Tuning Tab
        self.tab_tuning = QWidget()
        lay_tuning = QVBoxLayout(self.tab_tuning)
        lay_tuning.setContentsMargins(0, 4, 0, 0)
        lay_tuning.setSpacing(0)
        lay_tuning.addWidget(g_pipeline)
        lay_tuning.addStretch()
        self.tabs.addTab(self.tab_tuning, "Tuning")
        
        # Models Tab
        self.tab_models = QWidget()
        lay_models = QVBoxLayout(self.tab_models)
        lay_models.setContentsMargins(0, 4, 0, 0)
        lay_models.setSpacing(6)
        lay_models.addWidget(self.mbox_models)
        lay_models.addStretch()
        self.tabs.addTab(self.tab_models, "Models")
        
        # Advanced Tab
        self.tab_advanced = QWidget()
        lay_advanced = QVBoxLayout(self.tab_advanced)
        lay_advanced.setContentsMargins(0, 4, 0, 0)
        lay_advanced.setSpacing(0)
        lay_advanced.addWidget(g_advanced)
        lay_advanced.addStretch()
        self.tabs.addTab(self.tab_advanced, "Advanced")
        
        ctrl_lay.addWidget(self.tabs)

        ctrl_scroll.setWidget(ctrl_panel)
        self.workspace_splitter.addWidget(ctrl_scroll)

        # RIGHT PREVIEW & STATUS PANEL
        preview_panel = QFrame()
        preview_panel.setObjectName("previewCard")
        prev_lay = QVBoxLayout(preview_panel)
        prev_lay.setContentsMargins(1, 1, 1, 1)
        prev_lay.setSpacing(0)

        self.preview_widget = BeforeAfterPreview()
        self.preview_widget.roi_changed_cb = self._on_roi_changed

        # Viewport Toolbar (Sopra l'anteprima)
        viewport_toolbar_frame = QFrame()
        viewport_toolbar_frame.setObjectName("viewportToolbar")
        viewport_toolbar = QHBoxLayout(viewport_toolbar_frame)
        viewport_toolbar.setContentsMargins(10, 6, 10, 6)
        viewport_toolbar.setSpacing(8)
        
        self.chk_preview = QCheckBox("Preview ROI")
        self.chk_preview.setMinimumWidth(110)
        self.chk_preview.setChecked(False)
        self.chk_preview.toggled.connect(self._on_preview_toggled)
        viewport_toolbar.addWidget(self.chk_preview)
        
        self.btn_run_preview = QPushButton("Process ROI")
        self.btn_run_preview.setObjectName("primaryBtn")
        self.btn_run_preview.setEnabled(False)
        self.btn_run_preview.setFixedHeight(27)
        self.btn_run_preview.setStyleSheet("QPushButton { padding:0 10px; font-size:10px; }")
        self.btn_run_preview.clicked.connect(self._on_run_preview)
        viewport_toolbar.addWidget(self.btn_run_preview)
        
        self.lbl_roi_status = QLabel("Use checkbox or Right-click & drag.")
        self.lbl_roi_status.setStyleSheet("font-size: 11px; color: #8e8e93; font-style: italic;")
        self.lbl_roi_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        viewport_toolbar.addWidget(self.lbl_roi_status)
        
        viewport_toolbar.addStretch()
        
        # Zoom controls group
        zoom_lbl = QLabel("Zoom:")
        zoom_lbl.setStyleSheet("font-size: 11px; color: #8e8e93;")
        viewport_toolbar.addWidget(zoom_lbl)
        
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setFixedHeight(26)
        self.btn_zoom_fit.setFixedWidth(40)
        self.btn_zoom_fit.setStyleSheet("QPushButton { font-size: 10px; padding: 0px; min-height: 18px; font-weight: normal; }")
        self.btn_zoom_fit.clicked.connect(self.preview_widget.set_zoom_fit)
        
        self.btn_zoom_100 = QPushButton("100%")
        self.btn_zoom_100.setFixedHeight(26)
        self.btn_zoom_100.setFixedWidth(45)
        self.btn_zoom_100.setStyleSheet("QPushButton { font-size: 10px; padding: 0px; min-height: 18px; font-weight: normal; }")
        self.btn_zoom_100.clicked.connect(self.preview_widget.set_zoom_100)
        
        self.btn_zoom_200 = QPushButton("200%")
        self.btn_zoom_200.setFixedHeight(26)
        self.btn_zoom_200.setFixedWidth(45)
        self.btn_zoom_200.setStyleSheet("QPushButton { font-size: 10px; padding: 0px; min-height: 18px; font-weight: normal; }")
        self.btn_zoom_200.clicked.connect(self.preview_widget.set_zoom_200)
        
        viewport_toolbar.addWidget(self.btn_zoom_fit)
        viewport_toolbar.addWidget(self.btn_zoom_100)
        viewport_toolbar.addWidget(self.btn_zoom_200)
        
        prev_lay.addWidget(viewport_toolbar_frame)
        prev_lay.addWidget(self.preview_widget, 1)

        self.workspace_splitter.addWidget(preview_panel)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        control_width = min(390, max(290, target_w // 3))
        self.workspace_splitter.setSizes([control_width, max(360, target_w - control_width)])

        # Bottom Status Bar (Footer)
        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_card.setFixedHeight(48)
        sc_lay = QHBoxLayout(status_card)
        sc_lay.setContentsMargins(12, 2, 12, 2)
        sc_lay.setSpacing(12)

        # Status info on the left
        status_info = QHBoxLayout()
        status_info.setSpacing(6)
        stitle = QLabel("Status")
        stitle.setStyleSheet("color:#D6D6D9; font-size:9pt; font-weight:600;")
        self.lbl_status = QLabel("Ready — configure the pipeline or process a ROI")
        self.lbl_status.setStyleSheet("color:#94A3B8; font-size:9pt;")
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        status_info.addWidget(stitle)
        status_info.addWidget(self.lbl_status)
        sc_lay.addLayout(status_info)
        
        # Slim progress bar in the center
        self.pbar = AnimatedProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6)
        self.pbar.setMinimumWidth(80)
        self.pbar.setStyleSheet("QProgressBar { background:#2A2A2D; border:none; border-radius:3px; } QProgressBar::chunk { background:#5E9EFF; border-radius:3px; }")
        sc_lay.addWidget(self.pbar, 1)

        # Buttons on the right
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_run    = QPushButton("Run Full Image")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setFixedHeight(32)
        self.btn_run.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 14px; min-height: 20px; font-weight: bold; }")
        
        self.btn_import = QPushButton("Apply to Siril")
        self.btn_import.setObjectName("successBtn")
        self.btn_import.setEnabled(False)
        self.btn_import.setFixedHeight(32)
        self.btn_import.setStyleSheet("QPushButton { font-size:11px; padding:2px 14px; min-height:20px; }")
        
        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setFixedHeight(32)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 14px; min-height: 20px; }")
        
        self.btn_close  = QPushButton("Close")
        self.btn_close.setFixedHeight(32)
        self.btn_close.setStyleSheet("QPushButton { font-size: 11px; padding: 2px 14px; min-height: 20px; font-weight: normal; }")
        
        self.btn_run.clicked.connect(self._on_run)
        self.btn_import.clicked.connect(self._on_import)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_close.clicked.connect(self.close)
        
        btn_row.addStretch()
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_import)
        btn_row.addWidget(self.btn_close)
        sc_lay.addLayout(btn_row)
        
        main_lay.addWidget(status_card)

        # The splitter remains user-resizable after launch; preview receives
        # all surplus width while controls stay scrollable on compact screens.

    def _load_config_into_ui(self):
        cfg = self.config
        
        # Load active edition and set up the dynamic star reduction range first
        self._update_edition_buttons_style()
        edition_label = "Nano" if _CURRENT_EDITION == "nano" else "Pro"
        self.lbl_hero_title.setText(f"SyQon Parallax ({edition_label})")
        self._update_star_level_range()

        # Load active mode if Pro edition
        mode_val = cfg.get("mode", "aesthetics")
        self.combo_mode.blockSignals(True)
        if mode_val == "aesthetics":
            self.combo_mode.setCurrentIndex(1)
        else:
            self.combo_mode.setCurrentIndex(0)
        self.combo_mode.blockSignals(False)
        self._update_mode_visibility()

        self.chk_correct.setChecked(bool(cfg.get("correct", True)))
        self.spin_star.setValue(float(cfg.get("star_level", 3.0)))
        self.chk_sharpen.setChecked(bool(cfg.get("sharpen", True)))
        self._update_sharpen_range()
        self.spin_tile.setValue(int(cfg.get("tile_size", 512)))
        self.spin_overlap.setValue(int(cfg.get("overlap", 64)))
        self.spin_pad.setValue(int(cfg.get("pad", 96)))
        
        batch_val = str(cfg.get("batch_size", "Auto"))
        idx = self.combo_batch.findText(batch_val)
        if idx >= 0:
            self.combo_batch.setCurrentIndex(idx)
        else:
            self.combo_batch.setCurrentIndex(0)
        self.chk_mtf.setChecked(bool(cfg.get("use_mtf", True)))
        self.spin_mtf.setValue(float(cfg.get("mtf_target", 0.25)))
        self.chk_linked.setChecked(bool(cfg.get("linked_stretch", False)))
        mtf_on = self.chk_mtf.isChecked()
        self.spin_mtf.setEnabled(mtf_on)
        self.chk_linked.setEnabled(mtf_on)

        # Initialize sharpen control states
        sharpen_on = self.chk_sharpen.isChecked()
        self.sld_sharpen.setEnabled(sharpen_on)
        self.lbl_sharpen.setEnabled(sharpen_on)
        self.lbl_sharpen_strength.setEnabled(sharpen_on)

        self.chk_live_update.setChecked(bool(cfg.get("live_update", True)))

        # Dynamic preview stretch updates
        self.chk_mtf.toggled.connect(lambda _: self._update_preview())
        self.spin_mtf.valueChanged.connect(lambda _: self._update_preview())
        self.chk_linked.toggled.connect(lambda _: self._update_preview())

        # Render initial preview
        self._update_preview()
        
        # Force lock of performance/advanced settings at launch
        self.chk_unlock_advanced.setChecked(False)
        self._on_unlock_toggled(False)

    def _update_preview(self):
        cfg = self._gather_config()
        apply_stretch = cfg["use_mtf"]
        stretch_target = cfg["mtf_target"]
        linked = cfg["linked_stretch"]
        
        # Calculate MTF parameters on raw image downsampled
        h, w = self.preview_raw.shape[:2]
        max_dim = 800
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            step = int(1 / scale)
            if step < 1: step = 1
            preview_raw_ds = self.preview_raw[::step, ::step, :].copy()
        else:
            preview_raw_ds = self.preview_raw.copy()
            
        mtf_params = None
        if apply_stretch:
            mtf_params = (_mtf_params_linked(preview_raw_ds, targetbg=stretch_target)
                          if linked
                          else _mtf_params_unlinked(preview_raw_ds, targetbg=stretch_target))
        
        # Cache img_before stretch computation to keep GUI responsive during tile updates
        cache_key = (apply_stretch, stretch_target, linked)
        if getattr(self, "_img_before_cache_key", None) != cache_key or getattr(self, "_img_before_cached", None) is None:
            self._img_before_cached = _numpy_to_qimage(self.preview_raw, apply_stretch, stretch_target, linked, mtf_params)
            self._img_before_cache_key = cache_key

        img_before = self._img_before_cached
        img_after = None
        if self.processed_hwc is not None:
            img_after = _numpy_to_qimage(self.processed_hwc, apply_stretch, stretch_target, linked, mtf_params)
        elif self.processed_roi_hwc is not None and self.processed_roi_rect is not None:
            # Convert only the processed crop, then composite it over the cached
            # original QImage. This replaces repeated full-resolution NumPy
            # conversions during local ROI previews.
            roi_img = _numpy_to_qimage(
                self.processed_roi_hwc, apply_stretch, stretch_target, linked, mtf_params
            )
            img_after = img_before.copy()
            painter = QPainter(img_after)
            x, y, rw, rh = self.processed_roi_rect
            painter.drawImage(QRect(x, y, rw, rh), roi_img)
            painter.end()
            
        self.preview_widget.set_images(img_before, img_after)

    def _gather_config(self) -> dict:
        val = self.sld_sharpen.value() / 100.0
        if _CURRENT_EDITION == "nano":
            sharpen_alpha = val * (1.5 / 2.0)
            mode_str = "classic"
        else:
            sharpen_alpha = val
            mode_text = self.combo_mode.currentText().lower()
            mode_str = "aesthetics" if "defined" in mode_text else "classic"

        return {
            "edition":        _CURRENT_EDITION,
            "mode":           mode_str,
            "correct":        self.chk_correct.isChecked(),
            "star_level":     self.spin_star.value(),
            "sharpen":        self.chk_sharpen.isChecked(),
            "sharpen_alpha":  sharpen_alpha,
            "tile_size":      self.spin_tile.value(),
            "overlap":        self.spin_overlap.value(),
            "pad":            self.spin_pad.value(),
            "batch_size":     self.combo_batch.currentText(),
            "use_mtf":        self.chk_mtf.isChecked(),
            "mtf_target":     self.spin_mtf.value(),
            "linked_stretch": self.chk_linked.isChecked(),
            "use_gpu":        not self.chk_no_gpu.isChecked(),
            "live_update":     self.chk_live_update.isChecked(),
            "allow_large_live_preview": False,
        }

    def _on_sharpen_changed(self, val: int):
        self.lbl_sharpen.setText(f"{val/100:.2f}")
        if val > 200:
            self.sld_sharpen.setStyleSheet("""
                QSlider::sub-page:horizontal {
                    background: #ff453a;
                }
                QSlider::handle:horizontal:hover {
                    border-color: #ff453a;
                    background: #ff453a;
                }
            """)
            self.lbl_sharpen_warn.setText("Warning: Using a value greater than 2.0 makes the effect exaggerated and is recommended only and exclusively if there is strong deblur.")
        else:
            self.sld_sharpen.setStyleSheet("")
            self.lbl_sharpen_warn.setText("")

    def _refresh_model_status(self):
        # Update Classic models status
        for key, path_fn, label in [
            ("classic_correction",  lambda: _correction_path("classic"),  "Correction"),
            ("classic_star_reduce", lambda: _star_reduce_path("classic"),  "Sharpen Star"),
            ("classic_sharpen",     lambda: _sharpen_path("classic"),      "Sharpen"),
        ]:
            lbl = self._model_status[key]
            installed = path_fn().exists()
            color = '#30d158' if installed else '#ff453a'
            status_text = 'Installed' if installed else 'Missing'
            lbl.setText(f"<span style='color:{color}; font-size: 11px;'>●</span> <b>{label}:</b> <span style='color:{color}; font-size: 11px;'>{status_text}</span>")

        # Update Aesthetics models status (only if in Pro edition)
        if _CURRENT_EDITION == "pro":
            for key, path_fn, label in [
                ("aesthetics_correction",  lambda: _correction_path("aesthetics"),  "Correction (StarOnly)"),
                ("aesthetics_star_reduce", lambda: _star_reduce_path("aesthetics"),  "Sharpen Star"),
                ("aesthetics_sharpen",     lambda: _sharpen_path("aesthetics"),      "Sharpen (Deblur)"),
            ]:
                lbl = self._model_status[key]
                installed = path_fn().exists()
                color = '#30d158' if installed else '#ff453a'
                status_text = 'Installed' if installed else 'Missing'
                lbl.setText(f"<span style='color:{color}; font-size: 11px;'>●</span> <b>{label}:</b> <span style='color:{color}; font-size: 11px;'>{status_text}</span>")

    @staticmethod
    def _classify_pro_model_file(src_path: Path):
        """Map official Pro model names to one unambiguous pipeline slot."""
        stem = src_path.stem.lower()
        slots = (
            ("aesthetics_correction", ("a_starcorrect", "aesthetics_staronly", "staronly_base")),
            ("aesthetics_star_reduce", ("a_starreduction", "aesthetics_starreduction", "best_starreduction")),
            ("aesthetics_sharpen", ("a_deblur", "aesthetics_deblur", "deblur")),
            ("classic_correction", ("parallax_correction", "classic_correction")),
            ("classic_star_reduce", ("parallax_star_reduction", "classic_star_reduction")),
            ("classic_sharpen", ("parallax_sharpen", "classic_sharpen", "top_best")),
        )
        for slot, accepted_stems in slots:
            if any(token in stem for token in accepted_stems):
                return slot
        return None

    @staticmethod
    def _validate_pro_model_slot(src_path: Path, slot: str) -> str:
        """Strictly prove that a selected checkpoint belongs in its target slot."""
        loaders = {
            "classic_correction": (_load_correction_model, "classic"),
            "classic_star_reduce": (_load_star_reduce_model, "classic"),
            "classic_sharpen": (_load_sharpen_model, "classic"),
            "aesthetics_correction": (_load_correction_model, "aesthetics"),
            "aesthetics_star_reduce": (_load_star_reduce_model, "aesthetics"),
            "aesthetics_sharpen": (_load_sharpen_model, "aesthetics"),
        }
        loader, mode = loaders[slot]
        try:
            loader(str(src_path), torch.device("cpu"), mode)
            return ""
        except Exception as exc:
            return str(exc).splitlines()[0]
        finally:
            clear_model_cache()

    def _on_select_models_clicked(self):
        src_files, _ = QFileDialog.getOpenFileNames(
            self, "Select Parallax AI Models", "",
            "PyTorch model (*.pth *.pt);;All Files (*)"
        )
        if not src_files:
            return
            
        import shutil
        installed_count = 0
        errors = []

        for src in src_files:
            src_path = Path(src)
            if not src_path.exists():
                continue
                
            name_lower = src_path.name.lower()
            dest_name = None
            label = None
            
            if _CURRENT_EDITION == "nano":
                if "corrector" in name_lower or "correction" in name_lower:
                    dest_name = "f_nano_corrector.pt"
                    label = "Correction model"
                elif "reduce" in name_lower or "reduction" in name_lower:
                    dest_name = "f_nano_reduce.pth"
                    label = "Sharpen Star model"
                elif "sharp" in name_lower or "sharpen" in name_lower:
                    dest_name = "f_nano_sharp.pth"
                    label = "Sharpen model"
            else:  # Pro: two isolated, complete model families.
                slot = self._classify_pro_model_file(src_path)
                destinations = {
                    "classic_correction": ("parallax_correction.pth", "Classic Correction"),
                    "classic_star_reduce": ("parallax_star_reduction.pth", "Classic Sharpen Star"),
                    "classic_sharpen": ("parallax_sharpen.pth", "Classic Sharpen"),
                    "aesthetics_correction": ("aesthetics_staronly.pth", "Defined Correction (StarOnly)"),
                    "aesthetics_star_reduce": ("aesthetics_starreduction.pth", "Defined Sharpen Star"),
                    "aesthetics_sharpen": ("aesthetics_deblur.pth", "Defined Nebula Deblur"),
                }
                if slot is None:
                    errors.append(
                        f"File '{src_path.name}' is not an official Classic or Defined model name. "
                        "Select the six supplied Parallax model files without renaming them."
                    )
                    continue
                validation_error = self._validate_pro_model_slot(src_path, slot)
                if validation_error:
                    errors.append(
                        f"File '{src_path.name}' does not match the {destinations[slot][1]} architecture: "
                        f"{validation_error}"
                    )
                    continue
                dest_name, label = destinations[slot]
                    
            if dest_name is None:
                errors.append(f"File '{src_path.name}' is not recognized as a Parallax model.")
                continue
                
            dst = self.engine_dir / dest_name
            try:
                shutil.copy2(str(src_path), str(dst))
                installed_count += 1
            except Exception as e:
                errors.append(f"Failed to copy '{src_path.name}': {e}")
                
        clear_model_cache()
        self._refresh_model_status()
        self._update_preview()
        
        if errors:
            err_msg = "\n".join(errors)
            QMessageBox.warning(self, "Parallax Model Installation", 
                                f"Installed {installed_count} model(s) with some issues:\n\n{err_msg}")
        else:
            QMessageBox.information(self, "Parallax Model Installation", 
                                    f"Successfully installed all {installed_count} model(s)!")
            self.lbl_status.setText(f"Installed {installed_count} model(s) successfully.")

    def _remove_model(self, key: str):
        paths = {
            "classic_correction":      _correction_path("classic"),
            "classic_star_reduce":     _star_reduce_path("classic"),
            "classic_sharpen":         _sharpen_path("classic"),
            "aesthetics_correction":   _correction_path("aesthetics"),
            "aesthetics_star_reduce":  _star_reduce_path("aesthetics"),
            "aesthetics_sharpen":      _sharpen_path("aesthetics"),
        }
        p = paths[key]
        if not p.exists():
            return
        reply = QMessageBox.question(self, "Remove Model", f"Remove {p.name} from disk?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            p.unlink(missing_ok=True)
            clear_model_cache()
        except Exception as e:
            QMessageBox.warning(self, "Parallax", f"Remove failed: {e}")
        self._refresh_model_status()

    def _on_get_models_clicked(self):
        url = "https://syqon.eu/free-parallax-nano" if _CURRENT_EDITION == "nano" else PARALLAX_BUY_URL
        QDesktopServices.openUrl(url)

    # ---------------------------------------------------------------- run

    def _validate(self, cfg: dict) -> str:
        mode = cfg.get("mode", "classic")
        if not cfg["correct"] and cfg["star_level"] == 0 and cfg["sharpen_alpha"] <= 0.0:
            return "Enable at least one processing stage."
        if cfg["correct"] and not _correction_path(mode).exists():
            return "Correction model not installed."
        if cfg["star_level"] > 0 and not _star_reduce_path(mode).exists():
            return "Sharpen Star model not installed."
        if cfg["sharpen_alpha"] > 0 and not _sharpen_path(mode).exists():
            return "Sharpen model not installed."
        if cfg["overlap"] >= cfg["tile_size"]:
            return "Overlap must be smaller than tile size."
        return ""

    def _set_busy(self, busy: bool):
        self.btn_run.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_run_preview.setEnabled(not busy and self.preview_widget.roi_rect is not None)

    def _on_run(self):
        if self._worker_thread and self._worker_thread.isRunning():
            return
        cfg = self._gather_config()
        err = self._validate(cfg)
        if err:
            QMessageBox.warning(self, "SyQon Parallax", err)
            return
        if (self.preview_raw.shape[0] * self.preview_raw.shape[1] >= _LARGE_IMAGE_PIXELS
                and cfg.get("live_update", True)):
            prompt = QMessageBox(self)
            prompt.setIcon(QMessageBox.Icon.Warning)
            prompt.setWindowTitle("Large Image Detected")
            prompt.setText("This image is large enough that live preview can noticeably slow processing.")
            prompt.setInformativeText(
                "Disable live preview to preserve performance, or keep it enabled at your own risk."
            )
            disable_btn = prompt.addButton(
                "Disable Preview (Recommended)", QMessageBox.ButtonRole.AcceptRole
            )
            keep_btn = prompt.addButton(
                "Keep Preview (May Slow Processing)", QMessageBox.ButtonRole.RejectRole
            )
            prompt.setDefaultButton(disable_btn)
            prompt.exec()
            if prompt.clickedButton() is keep_btn:
                cfg["allow_large_live_preview"] = True
                self.lbl_status.setText("Live preview kept enabled for this large image.")
            else:
                cfg["live_update"] = False
                cfg["allow_large_live_preview"] = False
                self.chk_live_update.setChecked(False)
        self.config.update(cfg)
        save_config(self.config, self.siril)
        
        # Disable import and reset processed cache
        self.btn_import.setEnabled(False)
        self.processed_hwc = None
        self.processed_roi_hwc = None
        self.processed_roi_rect = None
        self.pending_siril_result = None
        self.preview_widget.set_images(self.preview_widget.img_before, None)
        
        self._set_busy(True)
        self.pbar.setValue(0)
        self.lbl_status.setText("Starting…")
        self.preview_widget.show_split = False

        self._worker = ParallaxWorker(
            self.raw, cfg, self.orig_dtype, self.scale, self.orig_was_mono, self.engine_dir, self.flip_vertical
        )
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.live_update.connect(self._on_live_update)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.finished_ok.connect(lambda *_: self._worker_thread.quit())
        self._worker.finished_err.connect(lambda *_: self._worker_thread.quit())
        self._worker_thread.finished.connect(lambda: setattr(self, "_worker_thread", None))
        self._worker_thread.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling…")

    def _on_progress(self, done: int, total: int, msg: str):
        if total > 0:
            self.pbar.setValue(int(100.0 * done / total))
        self.lbl_status.setText(msg)

    def _on_live_update(self, intermediate_hwc):
        self.processed_hwc = intermediate_hwc
        self.processed_roi_hwc = None
        self.processed_roi_rect = None
        self._update_preview()
        # set_images() already schedules a repaint via update().  Forcing a
        # synchronous repaint/processEvents loop flickers on embedded Linux Qt.

    def _on_preview_toggled(self, checked):
        if checked:
            if not self.preview_widget.roi_rect and self.preview_widget.img_before:
                img_w = self.preview_widget.img_before.width()
                img_h = self.preview_widget.img_before.height()
                rw = min(512, img_w // 2)
                rh = min(512, img_h // 2)
                rx = (img_w - rw) // 2
                ry = (img_h - rh) // 2
                self.preview_widget.roi_rect = QRect(rx, ry, rw, rh)
                self.preview_widget.update()
                self._on_roi_changed()
        else:
            self.preview_widget.roi_rect = None
            self.preview_widget.update()
            self._on_roi_changed()

    def _on_roi_changed(self):
        if self.preview_widget.roi_rect:
            r = self.preview_widget.roi_rect
            self.lbl_roi_status.setText(f"Area: {r.width()}x{r.height()} px")
            self.btn_run_preview.setEnabled(True)
            if not self.chk_preview.isChecked():
                self.chk_preview.blockSignals(True)
                self.chk_preview.setChecked(True)
                self.chk_preview.blockSignals(False)
        else:
            if self.chk_preview.isChecked() and self.preview_widget.img_before:
                img_w = self.preview_widget.img_before.width()
                img_h = self.preview_widget.img_before.height()
                rw = min(512, img_w // 2)
                rh = min(512, img_h // 2)
                rx = (img_w - rw) // 2
                ry = (img_h - rh) // 2
                self.preview_widget.roi_rect = QRect(rx, ry, rw, rh)
                self.preview_widget.update()
                self._on_roi_changed()
                return

            self.lbl_roi_status.setText("Use checkbox or Right-click & drag.")
            self.btn_run_preview.setEnabled(False)
            if self.chk_preview.isChecked():
                self.chk_preview.blockSignals(True)
                self.chk_preview.setChecked(False)
                self.chk_preview.blockSignals(False)

    def _on_run_preview(self):
        if not self.preview_widget.roi_rect or self.raw is None:
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return
            
        cfg = self._gather_config()
        err = self._validate(cfg)
        if err:
            QMessageBox.warning(self, "SyQon Parallax", err)
            return
            
        r = self.preview_widget.roi_rect
        img_h, img_w = self.preview_raw.shape[:2]
        
        roi_x = max(0, min(img_w - 1, r.x()))
        roi_y = max(0, min(img_h - 1, r.y()))
        roi_w = max(16, min(img_w - roi_x, r.width()))
        roi_h = max(16, min(img_h - roi_y, r.height()))
        
        cfg["roi"] = (roi_x, roi_y, roi_w, roi_h)
        cfg["is_preview"] = True
        # Fit ROI + context padding into one inference tile whenever practical.
        # A default 512 ROI with pad=96 previously expanded to 704 and became
        # four tiles per model; a temporary 704 tile processes it in one pass.
        preview_extent = max(roi_w, roi_h) + 2 * int(cfg.get("pad", 96))
        preview_tile = max(int(cfg.get("tile_size", 512)), math.ceil(preview_extent / 8) * 8)
        if preview_tile <= 1024:
            cfg["tile_size"] = preview_tile
            cfg["overlap"] = min(int(cfg.get("overlap", 64)), preview_tile - 8)
        # Intermediate ROI frames cost more UI/compositing time than they save;
        # the crop itself is small and returns quickly as a single final update.
        cfg["live_update"] = False
        self.processed_hwc = None
        self.processed_roi_hwc = None
        self.processed_roi_rect = None
        self._active_preview_roi = cfg["roi"]
        
        self._set_busy(True)
        self.pbar.setValue(0)
        self.lbl_status.setText("Running local preview…")
        self.preview_widget.show_split = False
        
        self._worker = ParallaxWorker(
            self.raw, cfg, self.orig_dtype, self.scale, self.orig_was_mono, self.engine_dir, self.flip_vertical
        )
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.live_update.connect(self._on_live_update)
        self._worker.finished_ok.connect(self._on_preview_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        self._worker.finished_ok.connect(lambda *_: self._worker_thread.quit())
        self._worker.finished_err.connect(lambda *_: self._worker_thread.quit())
        self._worker_thread.finished.connect(lambda: setattr(self, "_worker_thread", None))
        self._worker_thread.start()

    def _on_preview_finished_ok(self, result_for_siril, result_hwc):
        self._set_busy(False)
        self.lbl_status.setText("Preview process complete! Compare before and after inside the selected area.")
        self.preview_widget.show_split = True
        
        # Save only the processed crop; _update_preview composites it over the
        # cached original image without allocating a full float result.
        self.processed_hwc = None
        self.processed_roi_hwc = result_hwc
        self.processed_roi_rect = getattr(self, "_active_preview_roi", None)
        self.pending_siril_result = None
        self.btn_import.setEnabled(False)
        
        self._update_preview()

    def _on_finished_ok(self, result_for_siril, result_hwc):
        self._set_busy(False)
        self.lbl_status.setText("Processing complete! Drag the slider to compare Before & After.")
        self.preview_widget.show_split = True
        
        # Save results for manual import
        self.pending_siril_result = result_for_siril
        self.processed_hwc        = result_hwc
        self.processed_roi_hwc    = None
        self.processed_roi_rect   = None
        
        # Refresh preview to display both before and after
        self._update_preview()
        
        # Enable Import button
        self.btn_import.setEnabled(True)

    def _on_import(self):
        if self.pending_siril_result is None:
            return
        self.lbl_status.setText("Applying result to Siril…")
        try:
            self.siril.undo_save_state("SyQon Parallax")
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(self.pending_siril_result)
            print("SyQon Parallax: result successfully applied to Siril.")
            # Automatically close on import success
            self.close()
        except Exception as exc:
            self.lbl_status.setText(f"Import failed: {exc}")
            QMessageBox.critical(self, "SyQon Parallax", f"Import failed: {exc}")

    def _on_finished_err(self, err: str):
        self._set_busy(False)
        self.preview_widget.show_split = True
        if err == "__cancelled__":
            self.lbl_status.setText("Cancelled.")
            return
        QMessageBox.critical(self, "SyQon Parallax", f"Processing error:\n{err[:800]}")
        self.lbl_status.setText("Error.")

    def _on_run_diagnostic(self):
        diag = DiagnosticDialog(self)
        diag.exec()

    def _on_unlock_toggled(self, checked: bool):
        self.lbl_warning.setVisible(checked)
        self.spin_tile.setEnabled(checked)
        self.spin_overlap.setEnabled(checked)
        self.spin_pad.setEnabled(checked)
        # All other advanced controls remain independently available. The lock
        # intentionally protects only geometry parameters that can create seams
        # or excessive memory use.
        self.combo_batch.setEnabled(True)
        self.chk_no_gpu.setEnabled(True)
        self.chk_mtf.setEnabled(True)
        mtf_on = self.chk_mtf.isChecked()
        self.spin_mtf.setEnabled(mtf_on)
        self.chk_linked.setEnabled(mtf_on)

    def closeEvent(self, ev):
        if self._worker and self._worker_thread and self._worker_thread.isRunning():
            self._worker.cancel()
            self._worker_thread.wait(8000)
        super().closeEvent(ev)

# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SyQon Parallax — Official Siril Edition")
    parser.add_argument("--no-correct",   action="store_true",   help="Skip aberration correction")
    parser.add_argument("--star-level",   type=float, default=None, help="Star reduction level 0.0-7.0")
    parser.add_argument("--sharpen",      type=float, default=None, help="Sharpen alpha 0.0-1.0")
    parser.add_argument("--tile",         type=int,   default=None)
    parser.add_argument("--overlap",      type=int,   default=None)
    parser.add_argument("--pad",          type=int,   default=None)
    parser.add_argument("--mtf-target",   type=float, default=None)
    parser.add_argument("--no-mtf",       action="store_true")
    parser.add_argument("--linked",       action="store_true", help="Linked MTF stretch")
    parser.add_argument("--no-gpu",       action="store_true")
    parser.add_argument("--edition",      type=str,   default=None, choices=["nano", "pro"], help="Edition: nano or pro")
    args = parser.parse_args()

    siril = None
    try:
        siril = s.SirilInterface()
        siril.connect()
        print("Connected to Siril.")
    except Exception as exc:
        print(f"Could not connect to Siril: {exc}", file=sys.stderr)
        sys.exit(1)

    if not siril.is_image_loaded():
        print("Error: No image loaded.", file=sys.stderr)
        sys.exit(1)

    engine_dir = _resolve_engine_dir(siril)
    config     = load_config(siril)

    global _CURRENT_EDITION
    if args.edition is not None:
        config["edition"] = args.edition
    _CURRENT_EDITION = config.get("edition", "nano")

    if args.no_correct:                  config["correct"]        = False
    if args.star_level  is not None:
        max_lvl = 7.0 if _CURRENT_EDITION == "pro" else 5.0
        config["star_level"] = min(max(args.star_level, 0.0), max_lvl)
    if args.sharpen     is not None:
        val = args.sharpen
        if _CURRENT_EDITION == "nano":
            val = min(val, 2.0) * (1.5 / 2.0)
        config["sharpen_alpha"] = val
    if args.tile        is not None:     config["tile_size"]      = args.tile
    if args.overlap     is not None:     config["overlap"]        = args.overlap
    if args.pad         is not None:     config["pad"]            = args.pad
    if args.mtf_target  is not None:     config["mtf_target"]     = args.mtf_target
    if args.no_mtf:                      config["use_mtf"]        = False
    if args.linked:                      config["linked_stretch"] = True
    config["use_gpu"] = not args.no_gpu

    # ---- Detect FITS Orientation (Flip Vertical if bottom-up) ----
    flip_vertical = False
    try:
        filename = siril.get_image_filename().lower()
        if filename.endswith((".fits", ".fit", ".fts")):
            flip_vertical = True
            try:
                header = siril.get_image_fits_header()
                if isinstance(header, dict):
                    roworder = header.get("ROWORDER", "").strip().upper()
                    if roworder == "TOP-DOWN":
                        flip_vertical = False
            except Exception:
                pass
    except Exception:
        pass

    # ---- CLI mode ----
    is_cli = siril.is_cli() and len(sys.argv) > 1
    if is_cli:
        image = siril.get_image()
        raw   = image.data
        xrgb, orig_dtype, scale, orig_was_mono = _prepare_for_inference(raw, flip_vertical=flip_vertical)

        def _prog(done, total, msg):
            print(f"  [{int(100*done/max(total,1)):3d}%] {msg}", end="\r")

        result_hwc = process_image(
            xrgb,
            correct        = config["correct"],
            star_level     = config["star_level"],
            sharpen_alpha  = config["sharpen_alpha"] if config.get("sharpen", True) else 0.0,
            tile           = config["tile_size"],
            overlap        = config["overlap"],
            pad            = config["pad"],
            use_mtf        = config["use_mtf"],
            mtf_target     = config["mtf_target"],
            linked_stretch = config["linked_stretch"],
            use_gpu        = config["use_gpu"],
            correction_path  = str(_correction_path(config.get("mode", "classic"))),
            star_reduce_path = str(_star_reduce_path(config.get("mode", "classic"))),
            sharpen_path     = str(_sharpen_path(config.get("mode", "classic"))),
            progress_cb    = _prog,
            mode           = config.get("mode", "classic"),
            batch_size_cfg = config.get("batch_size", "Auto"),
        )
        print()

        if flip_vertical:
            result_hwc = np.flip(result_hwc, axis=0)

        if orig_was_mono:
            result_2d = result_hwc.mean(axis=2)
            result_for_siril = _restore_dtype(result_2d[np.newaxis, ...], orig_dtype, scale)
        else:
            result_for_siril = _restore_dtype(result_hwc.transpose(2, 0, 1), orig_dtype, scale)

        try:
            with siril.image_lock():
                siril.set_image_pixeldata(result_for_siril)
            print("SyQon Parallax: image updated in Siril.")
        except Exception as exc:
            print(f"Could not update Siril image: {exc}")
        return

    # ---- GUI mode ----
    app = QApplication.instance() or QApplication(sys.argv)

    image = siril.get_image()
    raw   = image.data
    xrgb, orig_dtype, scale, orig_was_mono = _prepare_for_inference(raw, flip_vertical=flip_vertical)

    gui = ParallaxGUI(siril, config, raw, orig_dtype, scale, orig_was_mono, engine_dir, flip_vertical=flip_vertical)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
