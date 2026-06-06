# (c) Adrian Knagg-Baugh 2026
# (c) SyQon / Franklin Marek 2026
# SPDX-License-Identifier: MIT
"""
    ███████╗██╗   ██╗ ██████╗  ██████╗ ███╗   ██╗
    ██╔════╝╚██╗ ██╔╝██╔═══██╗██╔═══██╗████╗  ██║
    ███████╗ ╚████╔╝ ██║   ██║██║   ██║██╔██╗ ██║
    ╚════██║  ╚██╔╝  ██║▄▄ ██║██║   ██║██║╚██╗██║
    ███████║   ██║   ╚██████╔╝╚██████╔╝██║ ╚████║
    ╚══════╝   ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═══╝
    ★  P R I S M   —   O F F I C I A L   E D I T I O N  ★
    ┌──────────────────────────────────────────────────┐
    │              Siril Edition v2.0                  │
    │    Advanced Denoising — Official SyQon Build     │
    │             https://syqon.it/prism               │
    └──────────────────────────────────────────────────┘

Official SyQon Prism script for Siril — self-contained single file edition.

All inference code is embedded directly; no separate download of
syqon_prism_inference.py is required. Model weights (.pt files) are
still downloaded / updated automatically on first run.

Usage:
    GUI mode (default):
        Add this script to the Scripts menu and run it. On first run the
        Prism Mini model will be downloaded automatically (~377 MB).

    Programmatic / CLI mode:
        pyscript SyQon-Prism.py --tile-size 512 --overlap 64 \\
            --pad 96 --modulation 1.0 --model deep --no-gpu

    CLI parameters:
        --tile-size    Tile size in pixels (default 512)
        --overlap      Tile overlap in pixels (default 64)
        --pad          Reflect-pad before tiling in pixels (default 96)
        --modulation   Blend 0.0 (original) – 1.0 (full denoise) (default 1.0)
        --model        mini (default) or deep
        --no-gpu       Disable GPU acceleration
        --force-update Force update check for model weights

Version History:
================
v1.0.0  Initial release
v1.0.1  Single image processing no longer saves a denoised_ prefix image, it just
        updates the image in Siril like other processes do.
v1.0.2  More granular error messages when setting up the inference module and torch
        model.
v1.0.3  Add PySide6 import in preparation to migrate to it.
v2.0.0  Official SyQon single-file release for Siril
        Improvements over the community Siril script:
        - Reflect-padded edges to eliminate border tile artefacts
        - Draggable before/after live preview updated tile-by-tile (single images)
        - Persistent JSON configuration
v2.0.1  If no CLI arguments, run in GUI mode by default
"""

from __future__ import annotations

# ============================================================================
# Standard-library imports (always available)
# ============================================================================

import argparse
import base64
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional, Tuple

# ============================================================================
# sirilpy bootstrap — installs / ensures dependencies
# ============================================================================

import sirilpy as s
from sirilpy.utility import download_with_progress

s.ensure_installed("PySide6", "astropy", "scipy")

# Intel Mac: keep numpy <2 for binary-compatibility
if platform.system() == "Darwin" and platform.machine() != "arm64":
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0",
                               "--quiet", "--exists-action", "i"])
    except subprocess.CalledProcessError as exc:
        print(f"Warning: could not pin numpy<2.0: {exc}")

# Ensure a PyTorch wheel is present
th = s.TorchHelper()
th.ensure_torch()

# Windows: fall back to DirectML if neither CUDA nor XPU is available
if sys.platform == "win32":
    try:
        import torch as _t
        if not _t.cuda.is_available() and not (hasattr(_t, "xpu") and _t.xpu.is_available()):
            s.ensure_installed("torch-directml")
    except Exception:
        pass

# ============================================================================
# Third-party imports (now guaranteed present)
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from scipy import stats as scipy_stats  # noqa: F401  (kept for inference parity)

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, QTimer, QEvent
from PySide6.QtGui import (QDesktopServices, QImage, QPainter, QPen, QColor,
                          QPixmap, QFont, QCursor, QLinearGradient, QBrush,
                          QPainterPath)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QSpinBox, QVBoxLayout, QWidget, QSlider, QSizePolicy,
    QFrame, QGroupBox,
)

# ============================================================================
# Script-level constants
# ============================================================================

SCRIPT_VERSION   = "2.0.1"
BASE_URL         = "https://siril.syqon.it"
PRISM_DEEP_URL   = "https://syqon.it/prism"

# Model weight files live next to the script in Siril's user-data dir.
# We resolve this after connecting to Siril (see _resolve_engine_dir).
_ENGINE_DIR: Path | None = None   # set by _resolve_engine_dir()

def _resolve_engine_dir(siril) -> Path:
    global _ENGINE_DIR
    if _ENGINE_DIR is not None:
        return _ENGINE_DIR
    try:
        user_dir = siril.get_siril_userdatadir() if siril else str(Path.home() / ".siril")
    except Exception:
        user_dir = str(Path.home() / ".siril")
    _ENGINE_DIR = Path(user_dir) / "syqon_prism"
    _ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    return _ENGINE_DIR

# Resolved lazily after Siril connects
def _mini_path() -> Path:
    return _ENGINE_DIR / "prism_mini.pt"

def _deep_path() -> Path:
    return _ENGINE_DIR / "prism_deep.pt"

# ============================================================================
# JSON Config
# ============================================================================

DEFAULT_CONFIG = {
    "tile_size":  512,
    "overlap":    96,
    "pad":        96,
    "modulation": 100,   # integer 0-100 (percent), matches slider
    "use_amp":    False,
    "model":      "Prism Mini",
    "ihs_target": 0.15,  # IHS stretch target median (_DEFAULT_TARGET)
}


def _config_path(siril) -> Path:
    try:
        cfg_dir = siril.get_siril_configdir() if siril else str(Path.home() / ".siril")
    except Exception:
        cfg_dir = str(Path.home() / ".siril")
    return Path(cfg_dir) / "syqon_prism_config.json"


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
# Network / download helpers  (mirrors community script exactly)
# ============================================================================

def _has_network() -> bool:
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0)
        try:
            sock.connect(("8.8.8.8", 80))
            return True
        finally:
            sock.close()
    except OSError:
        return False


def _download_file(url: str, dest: Path, desc: str, silent: bool = False) -> bool:
    if not silent:
        print(f"Downloading {desc} …")
    try:
        with urllib.request.urlopen(url) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)
        if not silent:
            print(f"  → {dest}")
        return True
    except Exception as exc:
        if not silent:
            print(f"Error downloading {desc}: {exc}", file=sys.stderr)
        return False


def _should_check_updates(engine_dir: Path, force: bool = False) -> bool:
    if not _has_network():
        print("Offline — skipping update check.")
        return False
    if force:
        return True
    stamp = engine_dir / "last_update.date"
    now   = datetime.now().strftime("%Y%m%d%H")
    if not stamp.exists():
        return True
    try:
        return now > stamp.read_text().strip()
    except Exception:
        return True


def _touch_stamp(engine_dir: Path) -> None:
    try:
        (engine_dir / "last_update.date").write_text(
            datetime.now().strftime("%Y%m%d%H")
        )
    except Exception:
        pass


def _remote_version(file_path: Path, file_name: str) -> Tuple[bool, Optional[str]]:
    """Returns (needs_update, remote_version_str | None)."""
    tmp = Path(str(file_path) + ".date.tmp")
    try:
        if not _download_file(f"{BASE_URL}/{file_name}.date", tmp,
                              f"{file_name} version info", silent=True):
            return False, None
        remote = tmp.read_text().strip()
        tmp.unlink(missing_ok=True)
        local_f = Path(str(file_path) + ".date")
        if not local_f.exists():
            return True, remote
        local = local_f.read_text().strip()
        if remote > local:
            print(f"Update available for {file_name}: local={local} remote={remote}")
            return True, remote
        print(f"{file_name} is up to date ({local}).")
        return False, remote
    except Exception as exc:
        print(f"Version check failed for {file_name}: {exc}")
        tmp.unlink(missing_ok=True)
        return False, None


def _write_version(file_path: Path, version: str) -> None:
    try:
        Path(str(file_path) + ".date").write_text(version)
    except Exception:
        pass


def _sha256_ok(file_path: Path, sha_file: Path) -> bool:
    try:
        expected = sha_file.read_text().strip().split()[0]
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == expected
    except Exception:
        return False


def _local_version(file_path: Path) -> Optional[str]:
    p = Path(str(file_path) + ".date")
    return p.read_text().strip() if p.exists() else None


def _already_verified(file_path: Path) -> bool:
    vf  = Path(str(file_path) + ".verified")
    cur = _local_version(file_path) or "unknown"
    return vf.exists() and vf.read_text().strip() == cur


def _mark_verified(file_path: Path) -> None:
    cur = _local_version(file_path) or "unknown"
    try:
        Path(str(file_path) + ".verified").write_text(cur)
    except Exception:
        pass


def _ask_update_qt(parent, file_name: str, local_v: Optional[str],
                   remote_v: str) -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle("Model Update Available")
    box.setText(f"A new version of {file_name} is available.")
    box.setInformativeText(
        f"Current: {local_v or 'Not installed'}\nNew: {remote_v}\n\n"
        "This file is approximately 380 MB.\nDownload now?"
    )
    box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    box.setDefaultButton(QMessageBox.StandardButton.No)
    return box.exec() == QMessageBox.StandardButton.Yes


# ============================================================================
# Model-weight management
# ============================================================================

def ensure_mini_model(engine_dir: Path, siril,
                      should_check: bool,
                      parent_widget=None) -> Path:
    """
    Ensure prism_mini.pt is present and up to date.
    Mirrors the community script's setup_model_torch() exactly.
    """
    model_file = engine_dir / "prism_mini.pt"
    sha_file   = engine_dir / "prism_mini.pt.sha256"
    file_name  = "prism_mini.pt"

    needs_update, remote_v = False, None
    if should_check:
        needs_update, remote_v = _remote_version(model_file, file_name)
    elif not model_file.exists():
        needs_update = True
        print(f"Prism Mini model not found, will download.")

    should_dl = False
    if not model_file.exists():
        print("Prism Mini model not present — downloading (~377 MB) …")
        should_dl = True
    elif needs_update:
        if parent_widget:
            should_dl = _ask_update_qt(parent_widget, file_name,
                                        _local_version(model_file), remote_v)
        else:
            should_dl = True
    else:
        print(f"Found Prism Mini model at {model_file}.")
        if not _already_verified(model_file) and sha_file.exists():
            if _sha256_ok(model_file, sha_file):
                _mark_verified(model_file)
            else:
                print("WARNING: Prism Mini checksum mismatch — model may be corrupted.")

    if should_dl:
        _download_file(f"{BASE_URL}/{file_name}.sha256", sha_file,
                       "SHA256 checksum", silent=True)
        download_with_progress(siril, f"{BASE_URL}/{file_name}", str(model_file))
        if sha_file.exists():
            if not _sha256_ok(model_file, sha_file):
                print("Checksum FAILED — removing corrupt download.", file=sys.stderr)
                model_file.unlink(missing_ok=True)
                sys.exit(1)
            print("Checksum OK.")
        if remote_v:
            _write_version(model_file, remote_v)
            _mark_verified(model_file)

    return model_file


# ============================================================================
# Reflect-pad helpers  (from SASpro _SyQonPrismProcessThread)
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


# ============================================================================
# Image preparation helpers
# ============================================================================

def _prepare_for_inference(data: np.ndarray):
    """
    Convert Siril pixel data to float32 HxWx3 [0,1] for the preview widget.

    Siril delivers images in planar format (C, H, W) with RGB channel order
    and row-0 at the bottom (FITS convention).  For display in Qt we need
    HxWx3 with row-0 at the top, so we flip vertically here.

    The flip is for display / preview only.  The raw data passed to the
    worker is untouched — process_image handles Siril's layout internally
    and returns results in the same layout, ready for set_image_pixeldata.

    Returns:
        xrgb        — HxWx3 float32 [0,1], vertically flipped for Qt display
        orig_dtype  — original numpy dtype
        scale       — scale factor (max of integer dtype, or 1.0 for float)
        orig_was_mono
    """
    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.integer):
        scale = float(np.iinfo(original_dtype).max)
        x = data.astype(np.float32) / scale
    else:
        x = data.astype(np.float32)
        scale = 1.0

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Detect Siril's planar (C, H, W) format
    is_planar = (
        x.ndim == 3
        and x.shape[0] in (1, 3, 4)
        and x.shape[0] < min(x.shape[1], x.shape[2])
    )

    if is_planar:
        x = np.transpose(x, (1, 2, 0))   # (C,H,W) → (H,W,C)

    orig_was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    if x.ndim == 2:
        xrgb = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        xrgb = np.repeat(x, 3, axis=2)
    else:
        xrgb = x[..., :3].copy()

    return xrgb.astype(np.float32, copy=False), original_dtype, scale, orig_was_mono


def _restore_dtype(data: np.ndarray,
                   original_dtype: np.dtype,
                   scale: float) -> np.ndarray:
    if original_dtype == np.float32:
        return data
    if np.issubdtype(original_dtype, np.integer):
        return np.clip(data * scale, 0, scale).astype(original_dtype)
    return data.astype(original_dtype)


# ============================================================================
# ── BEGIN EMBEDDED INFERENCE ENGINE ─────────────────────────────────────────
# (c) SyQon 2026  •  https://siril.syqon.it/LICENSE.pdf
# Adapted for single-file embedding; model paths resolved at runtime.
# ============================================================================

MAD_NORM = 1.48260222

# Logo (embedded from inference file — kept as-is)
LOGO_BASE64 = (
    "/9j/4AAQSkZJRgABAQIAHAAcAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH"
    "BwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQME"
    "BAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU"
    # (truncated for brevity — full string preserved from inference file)
    "FBQUFBT/wgARCA"
)

# ---------------------------------------------------------------------------
# Global model cache  (reset when GPU toggle changes)
# ---------------------------------------------------------------------------
_MODEL         = None
_DEVICE        = None
_AMP_ENABLED   = False
_RESIDUAL_MODE = True

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(use_gpu: Optional[bool] = True) -> torch.device:
    if not use_gpu:
        print("Using CPU")
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("Using CUDA (NVidia / AMD)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        if platform.machine() == "arm64":
            print("Using MPS (Apple Silicon)")
            return torch.device("mps")
        print("MPS detected but not applicable on Intel Mac — falling back to CPU")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Using XPU (Intel ARC)")
        return torch.device("xpu")
    try:
        import torch_directml
        if torch_directml.is_available():
            print("Using DirectML")
            return torch_directml.device()
    except ImportError:
        pass
    print("No GPU acceleration available — falling back to CPU")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Model architecture — NAFNet (mini + deep variants)
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFBlock_mini(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1  = LayerNorm2d(channels)
        self.conv1  = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.dwconv = nn.Conv2d(channels * 2, channels * 2, 3,
                                padding=1, groups=channels * 2, bias=True)
        self.sg     = SimpleGate()
        self.sca    = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.conv2  = nn.Conv2d(channels, channels, 1, bias=True)
        self.norm2  = LayerNorm2d(channels)
        self.ffn1   = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.ffn2   = nn.Conv2d(channels, channels, 1, bias=True)
        self.beta   = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma  = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sg(self.dwconv(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.sg(self.ffn1(self.norm2(x)))
        return x + self.ffn2(y) * self.gamma


class NAFNet_mini(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=48,
                 enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                 middle_blk_num=4, use_sigmoid=False):
        super().__init__()
        self.intro   = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending  = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock_mini(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2
        self.middle = nn.Sequential(*[NAFBlock_mini(ch) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=True), nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock_mini(ch) for _ in range(n)]))
        self.out_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for up, dec in zip(self.ups, self.decoders):
            x = up(x) + skips.pop(); x = dec(x)
        return self.out_act(self.ending(x))


class NAFBlock_deep(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1  = LayerNorm2d(channels)
        self.conv1  = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.dwconv = nn.Conv2d(channels * 2, channels * 2, 3,
                                padding=1, groups=channels * 2, bias=True)
        self.sg     = SimpleGate()
        self.ca     = ChannelAttention(channels)
        self.conv2  = nn.Conv2d(channels, channels, 1, bias=True)
        self.norm2  = LayerNorm2d(channels)
        self.ffn1   = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.ffn2   = nn.Conv2d(channels, channels, 1, bias=True)
        self.beta   = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma  = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sg(self.dwconv(self.conv1(self.norm1(x))))
        y = self.conv2(self.ca(y))
        x = x + y * self.beta
        y = self.sg(self.ffn1(self.norm2(x)))
        return x + self.ffn2(y) * self.gamma


class SmoothDownsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class SmoothUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2,
                                       mode="bilinear", align_corners=False))


class NAFNet_deep(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=48,
                 enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                 middle_blk_num=4):
        super().__init__()
        self.intro      = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending     = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
        self.encoders   = nn.ModuleList()
        self.downs      = nn.ModuleList()
        self.decoders   = nn.ModuleList()
        self.ups        = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock_deep(ch) for _ in range(n)]))
            self.downs.append(SmoothDownsample(ch, ch * 2))
            ch *= 2
        self.middle = nn.Sequential(*[NAFBlock_deep(ch) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(SmoothUpsample(ch, ch // 2))
            ch //= 2
            self.skip_convs.append(nn.Conv2d(ch * 2, ch, 1, bias=True))
            self.decoders.append(nn.Sequential(*[NAFBlock_deep(ch) for _ in range(n)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x   = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for up, sc, dec in zip(self.ups, self.skip_convs, self.decoders):
            x = up(x)
            s = skips.pop()
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:],
                                  mode="bilinear", align_corners=False)
            x = sc(torch.cat([x, s], dim=1))
            x = dec(x)
        return self.ending(x) + inp


def _create_model(ckpt: dict, model_is_mini: bool) -> nn.Module:
    args    = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    base_ch = int(args.get("base_ch", 48))
    depth   = int(args.get("depth",   4))
    print(f"Model config: base_ch={base_ch}, depth={depth}")
    depth   = max(1, min(8, depth))
    if depth >= 4:
        enc    = (2, 4, 6, 8) + (8,) * (depth - 4)
        dec    = (2,) * depth
        middle = 4
    elif depth == 3:
        enc, dec, middle = (2, 4, 6), (2, 2, 2), 3
    elif depth == 2:
        enc, dec, middle = (2, 4), (2, 2), 2
    else:
        enc, dec, middle = (2,), (2,), 2
    if model_is_mini:
        return NAFNet_mini(width=base_ch, enc_blk_nums=enc,
                           dec_blk_nums=dec, middle_blk_num=middle)
    return NAFNet_deep(width=base_ch, enc_blk_nums=enc,
                       dec_blk_nums=dec, middle_blk_num=middle)

# ---------------------------------------------------------------------------
# Model loading  (cached globally)
# ---------------------------------------------------------------------------

def load_model(use_gpu: Optional[bool] = True,
               model_path: Optional[Path] = None,
               ) -> Tuple[nn.Module, torch.device, bool, bool]:
    global _MODEL, _DEVICE, _AMP_ENABLED, _RESIDUAL_MODE

    if _MODEL is not None:
        return _MODEL, _DEVICE, _AMP_ENABLED, _RESIDUAL_MODE

    if model_path is None:
        model_path = _mini_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            "Please ensure the .pt file is in the syqon_prism directory."
        )

    print(f"Loading model from {model_path}")
    device      = get_device(use_gpu)
    ckpt        = torch.load(model_path, map_location="cpu", weights_only=True)
    model_is_mini = Path(model_path).stem == "prism_mini"

    model = _create_model(ckpt, model_is_mini).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    residual_mode = bool(ckpt.get("residual_output", True)) if isinstance(ckpt, dict) else True
    amp_enabled   = device.type in ("cuda", "mps")
    print(f"Residual mode: {residual_mode}")

    _MODEL         = model
    _DEVICE        = device
    _AMP_ENABLED   = amp_enabled
    _RESIDUAL_MODE = residual_mode
    return model, device, amp_enabled, residual_mode

# ---------------------------------------------------------------------------
# Per-channel Inverse Hyperbolic Stretch  (IHS)
# ---------------------------------------------------------------------------

_DEFAULT_B      = 6.0
_DEFAULT_TARGET = 0.15


def _to_chw(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    if data.ndim == 2:
        return data[np.newaxis], False
    if data.shape[2] <= 4:
        return data.transpose(2, 0, 1), True
    return data, False


def _to_hwc(chw: np.ndarray, was_hwc: bool, mono: bool) -> np.ndarray:
    if mono:
        return chw[0]
    return chw.transpose(1, 2, 0) if was_hwc else chw


def _adaptive_anchor(ch: np.ndarray) -> float:
    stride = max(1, ch.size // 2_000_000)
    sample = ch.flatten()[::stride]
    hist, edges = np.histogram(sample, bins=65536, range=(0.0, 1.0))
    smooth = np.convolve(hist, np.ones(50) / 50, mode='same')
    peak   = int(np.argmax(smooth))
    cands  = np.where(smooth[:peak] < smooth[peak] * 0.06)[0]
    anchor = float(edges[cands[-1]]) if len(cands) else float(np.percentile(sample, 0.5))
    return max(0.0, anchor)


def _ihs(x: np.ndarray, D: float, b: float) -> np.ndarray:
    num = np.arcsinh(D * x + b) - np.arcsinh(b)
    den = np.arcsinh(D + b)     - np.arcsinh(b)
    return num / (den if abs(den) > 1e-12 else 1e-6)


def _ihs_inverse(y: np.ndarray, D: float, b: float) -> np.ndarray:
    den = np.arcsinh(D + b) - np.arcsinh(b)
    if abs(den) < 1e-12:
        den = 1e-6
    return (np.sinh(y * den + np.arcsinh(b)) - b) / D


def _solve_log_d(sample: np.ndarray, target: float, b: float) -> float:
    valid = sample[sample > 1e-7]
    if valid.size == 0:
        return 2.0
    med = float(np.median(valid))
    if med < 1e-9:
        return 2.0
    lo, hi, best = 0.0, 7.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        D   = 10.0 ** mid
        val = float(_ihs(np.array([med]), D, b)[0])
        if abs(val - target) < 1e-4:
            best = mid; break
        lo, hi = (mid, hi) if val < target else (lo, mid)
    else:
        best = (lo + hi) / 2.0
    return best


def compute_ihs_params(data: np.ndarray,
                       target: float = _DEFAULT_TARGET,
                       b: float = _DEFAULT_B) -> dict:
    chw, _ = _to_chw(data.astype(np.float32))
    log_ds, anchors = [], []
    for i in range(chw.shape[0]):
        ch     = chw[i]
        anchor = _adaptive_anchor(ch)
        ch_anc = np.maximum(ch - anchor, 0.0)
        stride = max(1, ch_anc.size // 100_000)
        log_ds.append(_solve_log_d(ch_anc.flatten()[::stride], target, b))
        anchors.append(anchor)
    return {'log_d': log_ds, 'anchor': anchors, 'b': b}


def apply_ihs(data: np.ndarray, params: dict) -> np.ndarray:
    chw, was_hwc = _to_chw(data.astype(np.float32))
    mono = chw.shape[0] == 1
    b    = params['b']
    result = np.empty_like(chw)
    for i in range(chw.shape[0]):
        idx    = min(i, len(params['log_d']) - 1)
        D      = 10.0 ** params['log_d'][idx]
        anchor = params['anchor'][idx]
        ch_anc = np.maximum(chw[i] - anchor, 0.0)
        result[i] = np.clip(_ihs(ch_anc, D, b), 0.0, 1.0) * 0.995 + 0.005
    return _to_hwc(result, was_hwc, mono)


def apply_ihs_inverse(data: np.ndarray, params: dict) -> np.ndarray:
    chw, was_hwc = _to_chw(data.astype(np.float32))
    mono = chw.shape[0] == 1
    b    = params['b']
    result = np.empty_like(chw)
    for i in range(chw.shape[0]):
        idx    = min(i, len(params['log_d']) - 1)
        D      = 10.0 ** params['log_d'][idx]
        anchor = params['anchor'][idx]
        ch     = np.clip((chw[i] - 0.005) / 0.995, 0.0, 1.0)
        result[i] = np.clip(_ihs_inverse(ch, D, b) + anchor, 0.0, 1.0)
    return _to_hwc(result, was_hwc, mono)

# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _weight_1d(size: int, overlap: int, device: torch.device,
               dtype: torch.dtype) -> torch.Tensor:
    if overlap <= 0:
        return torch.ones(size, device=device, dtype=dtype)
    ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
    w = torch.ones(size, device=device, dtype=dtype)
    w[:overlap]  = ramp
    w[-overlap:] = torch.flip(ramp, dims=[0])
    return w


def _tile_positions(size: int, tile_size: int, stride: int) -> list:
    if size <= tile_size:
        return [0]
    pos = list(range(0, size - tile_size, stride))
    if pos[-1] != size - tile_size:
        pos.append(size - tile_size)
    return pos


def _edge_key(pos: int, size: int, tile: int) -> str:
    if pos == 0:             return "top"
    if pos + tile >= size:   return "bot"
    return "mid"


def _build_weight_cache(tile_size: int, overlap: int,
                        device: torch.device, dtype: torch.dtype) -> dict:
    def axis_w(sz, ov, dev, dt):
        base = _weight_1d(sz, ov, dev, dt)
        top, bot = base.clone(), base.clone()
        if ov > 0:
            top[:ov]  = 1.0
            bot[-ov:] = 1.0
        return top, base, bot

    wy_t, wy_m, wy_b = axis_w(tile_size, overlap, device, dtype)
    wx_l, wx_m, wx_r = axis_w(tile_size, overlap, device, dtype)
    cache = {}
    for yk, wy in (("top", wy_t), ("mid", wy_m), ("bot", wy_b)):
        for xk, wx in (("left", wx_l), ("mid", wx_m), ("right", wx_r)):
            cache[(yk, xk)] = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)
    return cache

# ---------------------------------------------------------------------------
# Core tiled inference  — patched to emit per-tile callbacks for live preview
# ---------------------------------------------------------------------------

def tile_inference_torch(
    model: nn.Module,
    x: torch.Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp_enabled: bool,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_start: int = 10,
    progress_range: int = 75,
    tile_callback: Optional[Callable[[int, int, int, int, np.ndarray], None]] = None,
) -> torch.Tensor:
    """
    Tiled inference.  tile_callback(y0, x0, ph, pw, patch_hwc_float32) fires
    after each tile — used by the live before/after preview widget.
    """
    _, _, h, w = x.shape

    # Whole-image fast path
    if tile_size <= 0 or tile_size >= min(h, w):
        if progress_callback:
            progress_callback(progress_start + progress_range // 2)
        with torch.no_grad(), amp.autocast(device_type=device.type, enabled=amp_enabled):
            result = model(x)
        if progress_callback:
            progress_callback(progress_start + progress_range)
        return result

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    ys    = _tile_positions(h, tile_size, stride)
    xs    = _tile_positions(w, tile_size, stride)
    total = len(ys) * len(xs)

    out    = torch.zeros_like(x, device=device)
    weight = torch.zeros((1, 1, h, w), device=device, dtype=x.dtype)
    cache  = _build_weight_cache(tile_size, overlap, device, x.dtype)
    xk_map = {"top": "left", "mid": "mid", "bot": "right"}

    count = 0
    with torch.no_grad():
        # Preview tiles should appear from the visual top row toward the right.
        # Because the preview QImage is vertically flipped for display, we walk
        # y positions in reverse here so the rendered progression starts at the
        # top-left corner from the user's perspective.
        for y in reversed(ys):
            yk = _edge_key(y, h, tile_size)
            for x0 in xs:
                xk  = xk_map[_edge_key(x0, w, tile_size)]
                w2  = cache[(yk, xk)]
                tile = x[:, :, y:y + tile_size, x0:x0 + tile_size]
                ph, pw = tile.shape[2], tile.shape[3]
                try:
                    with amp.autocast(device_type=device.type, enabled=amp_enabled):
                        pred = model(tile)
                except Exception:
                    pred = model(tile)

                out   [:, :, y:y + tile_size, x0:x0 + tile_size] += pred * w2
                weight[:, :, y:y + tile_size, x0:x0 + tile_size] += w2

                if tile_callback is not None:
                    # Send the properly blended result so far (not the raw tile)
                    blended_region = (
                        out[:, :, y:y + ph, x0:x0 + pw]
                        / weight[:, :, y:y + ph, x0:x0 + pw].clamp(min=1e-8)
                    )
                    patch_np = (
                        blended_region[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        .astype(np.float32, copy=False)
                    )
                    tile_callback(y, x0, ph, pw, patch_np)

                count += 1
                if progress_callback:
                    pct = progress_start + int((count / total) * progress_range)
                    progress_callback(pct)

    return out / weight.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# High-level process_image()
# ---------------------------------------------------------------------------

def process_image(
    img: np.ndarray,
    tile: int = 512,
    overlap: int = 64,
    use_amp: bool = True,
    use_gpu: Optional[bool] = True,
    modulation: float = 1.0,
    model_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    tile_callback: Optional[Callable[[int, int, int, int, np.ndarray], None]] = None,
    ihs_target: float = _DEFAULT_TARGET,
) -> np.ndarray:
    """
    Denoise a single image using tiled NAFNet + IHS autostretch.

    Identical semantics to the inference file's process_image() with the
    addition of tile_callback for live tile-by-tile preview updates.
    """
    def _emit(v):
        if progress_callback:
            progress_callback(v)

    _emit(0)

    if model_path is None:
        model_path = _mini_path()

    model, device, amp_en, residual_mode = load_model(
        use_gpu=use_gpu, model_path=model_path
    )
    amp_en = use_amp and amp_en

    # --- layout detection ---
    is_planar = False
    is_mono   = False
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        if img.shape[0] < min(img.shape[1], img.shape[2]):
            is_planar = True
            is_mono   = img.shape[0] == 1

    if is_planar:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        is_mono = True
        img = img[..., None]

    ihs_params = compute_ihs_params(img, target=ihs_target)

    if img.shape[2] == 1:
        is_mono = True
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    img_original = img.copy()
    img          = apply_ihs(img, ihs_params)
    _emit(5)

    print("Inferencing…")
    x = (
        torch.from_numpy(img.astype(np.float32))
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    _emit(10)

    pred = tile_inference_torch(
        model, x, tile, overlap, device, amp_en,
        progress_callback=progress_callback,
        progress_start=10,
        progress_range=75,
        tile_callback=tile_callback,
    )
    _emit(87)

    result = pred.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    result = apply_ihs_inverse(result, ihs_params)

    if modulation < 1.0:
        result = result * modulation + img_original * (1.0 - modulation)

    if is_mono:
        result = np.mean(result, axis=2, keepdims=True)
    if is_planar:
        result = np.transpose(result, (2, 0, 1))
    if is_mono and not is_planar:
        result = result.squeeze(-1)

    _emit(100)
    print(f"Done. mono={is_mono}, planar={is_planar}")
    return result

# ============================================================================
# ── END EMBEDDED INFERENCE ENGINE ───────────────────────────────────────────
# ============================================================================


# ============================================================================
# Qt worker thread
# ============================================================================

class PrismWorker(QObject):
    """Runs process_image() in a QThread, emitting signals for the GUI."""

    progress     = Signal(int)                        # 0-100
    tile_done    = Signal(int, int, int, int, object) # y0, x0, ph, pw, patch (padded coords)
    # finished_ok carries (denoised_for_siril, denoised_padded_rgb3)
    # denoised_for_siril   → in original Siril layout/dtype (ready for set_image_pixeldata)
    # denoised_padded_rgb3 → HxWx3 float32 [0,1] for the preview widget
    finished_ok  = Signal(object, object)
    finished_err = Signal(str)

    def __init__(self, raw: np.ndarray, tile: int, overlap: int,
                 use_amp: bool, use_gpu: bool,
                 modulation: float, model_path: Path,
                 orig_dtype: np.dtype, scale: float,
                 orig_was_mono: bool,
                 pad: int = 0,
                 ihs_target: float = _DEFAULT_TARGET):
        super().__init__()
        self.raw          = raw            # original Siril data — untouched
        self.tile         = tile
        self.overlap      = overlap
        self.use_amp      = use_amp
        self.use_gpu      = use_gpu
        self.modulation   = modulation
        self.model_path   = model_path
        self.orig_dtype   = orig_dtype
        self.scale        = scale
        self.orig_was_mono = orig_was_mono
        self.pad          = int(max(0, pad))
        self.ihs_target   = float(ihs_target)
        self._cancel      = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # --- Convert raw Siril data to HxWx3 float32 [0,1] for padding/inference ---
            xrgb, _, _, _ = _prepare_for_inference(self.raw)

            padded, orig_hw = _pad_reflect(xrgb, self.pad)

            def _tile_cb(y0, x0, ph, pw, patch):
                # Raw padded-space coords — preview widget is sized to match.
                if self._cancel:
                    raise InterruptedError("Cancelled")
                self.tile_done.emit(y0, x0, ph, pw, patch)

            def _prog(v):
                if self._cancel:
                    raise InterruptedError("Cancelled")
                self.progress.emit(v)

            # process_image expects HxWx3 float32 [0,1] — padded already is.
            denoised_padded = process_image(
                padded,
                tile              = self.tile,
                overlap           = self.overlap,
                use_amp           = self.use_amp,
                use_gpu           = self.use_gpu,
                modulation        = self.modulation,
                model_path        = self.model_path,
                progress_callback = _prog,
                tile_callback     = _tile_cb,
                ihs_target        = self.ihs_target,
            )

            # Ensure HxWx3 for the padded preview result
            if denoised_padded.ndim == 2:
                denoised_padded_rgb = np.stack([denoised_padded] * 3, axis=-1)
            elif denoised_padded.ndim == 3 and denoised_padded.shape[2] == 1:
                denoised_padded_rgb = np.repeat(denoised_padded, 3, axis=2)
            else:
                denoised_padded_rgb = denoised_padded[..., :3]

            # Unpad → HxWx3 float32
            denoised_hwc3 = _unpad(denoised_padded_rgb, orig_hw, self.pad)

            # Collapse to mono and restore original dtype.
            # Siril expects planar (1,H,W) for mono, RGB planar (3,H,W) for colour.
            if self.orig_was_mono:
                denoised_2d = denoised_hwc3.mean(axis=2)           # (H,W)
                denoised_2d_restored = _restore_dtype(denoised_2d, self.orig_dtype, self.scale)
                denoised_for_siril   = denoised_2d_restored[np.newaxis, ...]  # (1,H,W)
            else:
                denoised_chw = denoised_hwc3.transpose(2, 0, 1)    # (3,H,W) RGB
                denoised_for_siril = _restore_dtype(denoised_chw, self.orig_dtype, self.scale)

            self.finished_ok.emit(denoised_for_siril, denoised_padded_rgb)

        except InterruptedError:
            self.finished_err.emit("__cancelled__")
        except Exception as exc:
            import traceback
            self.finished_err.emit(f"{exc}\n{traceback.format_exc()}")


# ============================================================================
# Before / After split-preview widget
# ============================================================================

def _ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """HxWx3 float32 [0,1] → QImage RGB888, vertically flipped for display."""
    rgb8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb8 = np.ascontiguousarray(rgb8[::-1])
    h, w = rgb8.shape[:2]
    return QImage(rgb8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class SplitPreviewWidget(QWidget):
    """
    Full-image before/after preview with a draggable vertical divider.
    The after image is updated tile-by-tile during inference.
    Supports mouse-wheel zoom, right-click pan, and double-click reset.
    """

    zoom_changed = Signal(float)   # emitted when zoom level changes
    preview_requested = Signal()

    def __init__(self, before_rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self._before      = before_rgb.copy()
        self._after       = before_rgb.copy()
        self._before_pm:  Optional[QPixmap] = None
        self._after_pm:   Optional[QPixmap] = None
        self._split_frac: float = 0.5
        self._split_active: bool = False   # divider hidden until tiling completes
        self._split_reveal: float = 0.0
        self._has_after_updates: bool = False
        self._dragging:   bool  = False
        # Zoom & pan state
        self._zoom:       float = 1.0
        self._pan_x:      float = 0.0
        self._pan_y:      float = 0.0
        self._panning:    bool  = False
        self._pan_start_x: float = 0.0
        self._pan_start_y: float = 0.0
        self._pan_origin_x: float = 0.0
        self._pan_origin_y: float = 0.0
        self._mm_rect = None           # (x, y, w, h) set by paintEvent
        self._minimap_dragging = False  # True when dragging inside minimap
        self._split_timer = QTimer(self)
        self._split_timer.setInterval(16)
        self._split_timer.timeout.connect(self._advance_split_reveal)
        self._draw_rect = None
        self._selection_visible = False
        self._selection_dragging = False
        self._selection_size = int(min(512, before_rgb.shape[0], before_rgb.shape[1]))
        self._selection_cx = before_rgb.shape[1] / 2.0
        self._selection_cy = before_rgb.shape[0] / 2.0
        self._selection_dx = 0.0
        self._selection_dy = 0.0
        self._selection_screen_rect = None
        self._preview_btn = QPushButton("Run Preview", self)
        self._preview_btn.setVisible(False)
        self._preview_btn.setFixedHeight(26)
        self._preview_btn.clicked.connect(self.preview_requested.emit)
        self._rebuild_pixmaps()

    def set_preview_button_enabled(self, enabled: bool):
        self._preview_btn.setEnabled(enabled)

    def set_selection_visible(self, visible: bool):
        self._selection_visible = bool(visible)
        self._preview_btn.setVisible(self._selection_visible)
        self.update()

    def reset_selection(self, size: Optional[int] = None):
        h, w = self._before.shape[:2]
        if size is not None:
            self._selection_size = int(min(size, h, w))
        else:
            self._selection_size = int(min(self._selection_size, h, w))
        self._selection_cx = w / 2.0
        self._selection_cy = h / 2.0
        self.update()

    def focus_selection(self):
        if self._draw_rect is None:
            return
        x0, y0, sw, sh = self.get_selection_rect()
        W, H = self.width(), self.height()
        pm_w = self._before_pm.width() if self._before_pm else 1
        pm_h = self._before_pm.height() if self._before_pm else 1
        base_scale = min(W / max(pm_w, 1), H / max(pm_h, 1))
        target_zoom = min(
            max(1.0, 0.72 * W / max(sw * base_scale, 1e-6)),
            max(1.0, 0.72 * H / max(sh * base_scale, 1e-6)),
            20.0,
        )
        self._zoom = float(np.clip(target_zoom, 1.0, 20.0))
        dw = pm_w * base_scale * self._zoom
        dh = pm_h * base_scale * self._zoom
        sel_cx = x0 + sw / 2.0
        sel_cy = y0 + sh / 2.0
        self._pan_x = -(sel_cx / max(self._before.shape[1], 1) * dw - W / 2.0) - (W - dw) / 2.0
        # Preview drawing is vertically flipped, so centre on the mirrored Y.
        sel_cy_draw = self._before.shape[0] - sel_cy
        self._pan_y = -(sel_cy_draw / max(self._before.shape[0], 1) * dh - H / 2.0) - (H - dh) / 2.0
        self.zoom_changed.emit(self._zoom)
        self.update()

    def get_selection_rect(self) -> tuple[int, int, int, int]:
        h, w = self._before.shape[:2]
        size = int(np.clip(self._selection_size, 32, min(h, w)))
        half = size / 2.0
        cx = float(np.clip(self._selection_cx, half, w - half))
        cy = float(np.clip(self._selection_cy, half, h - half))
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x0 = int(np.clip(x0, 0, max(0, w - size)))
        y0 = int(np.clip(y0, 0, max(0, h - size)))
        return x0, y0, size, size

    def _screen_to_image(self, sx: float, sy: float):
        if self._draw_rect is None:
            return None
        ox, oy, dw, dh = self._draw_rect
        if dw <= 0 or dh <= 0:
            return None
        ix = (sx - ox) / dw * self._before.shape[1]
        # The displayed QImage is vertically flipped in _ndarray_to_qimage(),
        # so convert back from screen space to top-origin image space here.
        iy = (1.0 - ((sy - oy) / dh)) * self._before.shape[0]
        return ix, iy

    def _image_rect_to_screen_rect(self, x0: float, y0: float, w0: float, h0: float):
        if self._draw_rect is None:
            return None
        ox, oy, dw, dh = self._draw_rect
        img_w = max(float(self._before.shape[1]), 1.0)
        img_h = max(float(self._before.shape[0]), 1.0)
        rx = ox + (x0 / img_w) * dw
        # Mirror Y because the preview pixmap is drawn upside-down for Qt.
        ry = oy + ((img_h - (y0 + h0)) / img_h) * dh
        rw = (w0 / img_w) * dw
        rh = (h0 / img_h) * dh
        return rx, ry, rw, rh

    def _update_preview_button_position(self):
        if not self._selection_visible or self._selection_screen_rect is None:
            self._preview_btn.hide()
            return
        rx, ry, rw, rh = self._selection_screen_rect
        hint = self._preview_btn.sizeHint()
        bw = max(110, hint.width())
        self._preview_btn.resize(bw, 26)
        bx = int(np.clip(rx + rw / 2 - bw / 2, 8, max(8, self.width() - bw - 8)))
        by_below = int(ry + rh + 8)
        by = by_below
        if by + self._preview_btn.height() > self.height() - 8:
            by = int(max(8, ry - self._preview_btn.height() - 8))
        self._preview_btn.move(bx, by)
        self._preview_btn.show()

    def _advance_split_reveal(self):
        self._split_reveal = min(1.0, self._split_reveal + 0.08)
        self.update()
        if self._split_reveal >= 1.0:
            self._split_timer.stop()

    def _rebuild_pixmaps(self):
        self._before_pm = QPixmap.fromImage(_ndarray_to_qimage(self._before))
        self._after_pm  = QPixmap.fromImage(_ndarray_to_qimage(self._after))
        self.update()

    def update_tile(self, y0: int, x0: int, ph: int, pw: int,
                    tile_rgb: np.ndarray):
        H, W = self._after.shape[:2]
        y1, x1 = min(y0 + ph, H), min(x0 + pw, W)
        tile_rgb = np.asarray(tile_rgb)
        if tile_rgb.ndim == 2:
            tile_rgb = np.stack([tile_rgb] * 3, axis=-1)
        self._after[y0:y1, x0:x1, :] = tile_rgb[:y1 - y0, :x1 - x0, :]
        self._has_after_updates = True
        self._after_pm = QPixmap.fromImage(_ndarray_to_qimage(self._after))
        self.update()

    def activate_split(self):
        """Show the before/after divider once the tiled result is complete."""
        if self._split_active:
            return
        self._split_active = True
        self._split_reveal = 0.0
        self._split_timer.start()
        self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        self.update()

    def reset_after(self, after_rgb: np.ndarray):
        after_rgb = np.asarray(after_rgb)
        if after_rgb.ndim == 2:
            after_rgb = np.stack([after_rgb] * 3, axis=-1)
        self._after    = after_rgb.copy()
        self._after_pm = QPixmap.fromImage(_ndarray_to_qimage(self._after))
        self.update()

    def reset_before(self, before_rgb: np.ndarray):
        """Replace the before (left) image only — does not touch the after buffer."""
        before_rgb = np.asarray(before_rgb)
        if before_rgb.ndim == 2:
            before_rgb = np.stack([before_rgb] * 3, axis=-1)
        self._before    = before_rgb.copy()
        self._before_pm = QPixmap.fromImage(_ndarray_to_qimage(self._before))
        self.update()

    def reset_both(self, img_rgb: np.ndarray):
        """Set before and after to the same image — call at run-start to
        establish a consistent padded baseline before tiles paint in.
        Hides the split divider until tiling completes."""
        img_rgb = np.asarray(img_rgb)
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        # Hide divider until tiling completes
        self._split_timer.stop()
        self._split_active = False
        self._split_reveal = 0.0
        self._has_after_updates = False
        self._split_frac   = 0.5
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self._before    = img_rgb.copy()
        self._after     = img_rgb.copy()
        self._before_pm = QPixmap.fromImage(_ndarray_to_qimage(self._before))
        self._after_pm  = QPixmap.fromImage(_ndarray_to_qimage(self._after))
        self.update()

    # ---- Zoom helpers ----

    def set_zoom(self, zoom: float):
        """Set zoom level (1.0 = fit, clamped to 0.25–20.0)."""
        self._zoom = float(np.clip(zoom, 0.25, 20.0))
        self.zoom_changed.emit(self._zoom)
        self.update()

    def reset_zoom(self):
        """Reset zoom to fit and centre the image."""
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self):
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self._zoom / 1.25)

    # ---- Qt events ----

    def _minimap_click_to_pan(self, mx, my):
        """Translate a click at (mx, my) inside the minimap to pan values."""
        mm_x, mm_y, mm_w, mm_h = self._mm_rect
        W, H = self.width(), self.height()
        pm_w = self._before_pm.width() if self._before_pm else 1
        pm_h = self._before_pm.height() if self._before_pm else 1
        base_scale = min(W / max(pm_w, 1), H / max(pm_h, 1))
        dw = pm_w * base_scale * self._zoom
        dh = pm_h * base_scale * self._zoom
        # Normalised position within minimap [0..1]
        frac_x = (mx - mm_x) / max(mm_w, 1)
        frac_y = (my - mm_y) / max(mm_h, 1)
        frac_x = float(np.clip(frac_x, 0, 1))
        frac_y = float(np.clip(frac_y, 0, 1))
        # Pan so that frac corresponds to the centre of the viewport
        self._pan_x = -(frac_x * dw - W / 2.0) - (W - dw) / 2.0
        self._pan_y = -(frac_y * dh - H / 2.0) - (H - dh) / 2.0
        self.update()

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.LeftButton
                and self._selection_visible
                and self._selection_screen_rect is not None):
            rx, ry, rw, rh = self._selection_screen_rect
            px, py = ev.position().x(), ev.position().y()
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
                img_pt = self._screen_to_image(px, py)
                if img_pt is not None:
                    self._selection_dragging = True
                    self._selection_dx = img_pt[0] - self._selection_cx
                    self._selection_dy = img_pt[1] - self._selection_cy
                    self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                    return
        # Check minimap click first (left button only)
        if (ev.button() == Qt.MouseButton.LeftButton
                and self._mm_rect is not None):
            mm_x, mm_y, mm_w, mm_h = self._mm_rect
            px, py = ev.position().x(), ev.position().y()
            if mm_x <= px <= mm_x + mm_w and mm_y <= py <= mm_y + mm_h:
                self._minimap_dragging = True
                self._minimap_click_to_pan(px, py)
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return
        if ev.button() == Qt.MouseButton.LeftButton and self._split_active:
            self._dragging = True
        elif ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = True
            self._pan_start_x = ev.position().x()
            self._pan_start_y = ev.position().y()
            self._pan_origin_x = self._pan_x
            self._pan_origin_y = self._pan_y
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._selection_dragging:
                self._selection_dragging = False
                self.setCursor(QCursor(
                    Qt.CursorShape.SizeHorCursor if self._split_active
                    else Qt.CursorShape.ArrowCursor
                ))
                return
            if self._minimap_dragging:
                self._minimap_dragging = False
                self.setCursor(QCursor(
                    Qt.CursorShape.SizeHorCursor if self._split_active
                    else Qt.CursorShape.ArrowCursor
                ))
                return
            self._dragging = False
        elif ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self.setCursor(QCursor(
                Qt.CursorShape.SizeHorCursor if self._split_active
                else Qt.CursorShape.ArrowCursor
            ))

    def mouseMoveEvent(self, ev):
        if self._selection_dragging:
            img_pt = self._screen_to_image(ev.position().x(), ev.position().y())
            if img_pt is not None:
                self._selection_cx = img_pt[0] - self._selection_dx
                self._selection_cy = img_pt[1] - self._selection_dy
                self.update()
            return
        if self._minimap_dragging and self._mm_rect is not None:
            self._minimap_click_to_pan(ev.position().x(), ev.position().y())
            return
        if self._dragging:
            self._split_frac = float(
                np.clip(ev.position().x() / max(self.width(), 1), 0.02, 0.98)
            )
            self.update()
        elif self._panning:
            dx = ev.position().x() - self._pan_start_x
            dy = ev.position().y() - self._pan_start_y
            self._pan_x = self._pan_origin_x + dx
            self._pan_y = self._pan_origin_y + dy
            self.update()

    def mouseDoubleClickEvent(self, ev):
        """Double-click resets zoom to fit."""
        self.reset_zoom()

    def wheelEvent(self, ev):
        """Mouse wheel zooms in/out centred on cursor position."""
        old_zoom = self._zoom
        delta = ev.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = float(np.clip(old_zoom * factor, 0.25, 20.0))
        if new_zoom == old_zoom:
            return

        # Adjust pan so the point under the cursor stays fixed
        mx = ev.position().x()
        my = ev.position().y()
        W, H = self.width(), self.height()
        pm_w = self._before_pm.width() if self._before_pm else 1
        pm_h = self._before_pm.height() if self._before_pm else 1
        base_scale = min(W / max(pm_w, 1), H / max(pm_h, 1))

        # Old image origin
        old_dw = pm_w * base_scale * old_zoom
        old_dh = pm_h * base_scale * old_zoom
        old_ox = (W - old_dw) / 2.0 + self._pan_x
        old_oy = (H - old_dh) / 2.0 + self._pan_y

        # Relative cursor position within the image [0..1]
        rx = (mx - old_ox) / old_dw if old_dw else 0.5
        ry = (my - old_oy) / old_dh if old_dh else 0.5

        # New image size
        new_dw = pm_w * base_scale * new_zoom
        new_dh = pm_h * base_scale * new_zoom

        # New origin needed to keep (rx, ry) under cursor
        new_ox = mx - rx * new_dw
        new_oy = my - ry * new_dh

        self._pan_x = new_ox - (W - new_dw) / 2.0
        self._pan_y = new_oy - (H - new_dh) / 2.0
        self._zoom = new_zoom
        self.zoom_changed.emit(self._zoom)
        self.update()

    def paintEvent(self, ev):
        if self._before_pm is None or self._after_pm is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H    = self.width(), self.height()
        pm_w, pm_h = self._before_pm.width(), self._before_pm.height()
        base_scale = min(W / max(pm_w, 1), H / max(pm_h, 1))
        scale   = base_scale * self._zoom
        dw, dh  = int(pm_w * scale), int(pm_h * scale)
        ox      = int((W - dw) / 2.0 + self._pan_x)
        oy      = int((H - dh) / 2.0 + self._pan_y)
        self._draw_rect = (ox, oy, dw, dh)

        if self._split_active:
            split_x = int(W * self._split_frac)

            # Use the processed image as the base layer, then fade the left-side
            # "before" panel in when the compare becomes available.
            reveal = max(0.0, min(1.0, self._split_reveal))
            painter.drawPixmap(ox, oy, dw, dh, self._after_pm)
            painter.setClipRect(0, 0, split_x, H)
            painter.setOpacity(reveal)
            painter.drawPixmap(ox, oy, dw, dh, self._before_pm)
            painter.setOpacity(1.0)
            painter.setClipping(False)

            # --- Divider line with glow ---
            # Outer glow
            glow_alpha = int(40 * reveal)
            core_alpha = int(220 * reveal)
            painter.setPen(QPen(QColor(64, 200, 224, glow_alpha), 6, Qt.PenStyle.SolidLine))
            painter.drawLine(split_x, 0, split_x, H)
            # Core line
            painter.setPen(QPen(QColor(64, 200, 224, core_alpha), 2, Qt.PenStyle.SolidLine))
            painter.drawLine(split_x, 0, split_x, H)

            # --- Handle pill (centre of divider) ---
            handle_h, handle_w = 48, 22
            hx = split_x - handle_w // 2
            hy = H // 2 - handle_h // 2
            handle_path = QPainterPath()
            handle_path.addRoundedRect(float(hx), float(hy),
                                       float(handle_w), float(handle_h), 11, 11)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(20, 20, 25, int(210 * reveal))))
            painter.drawPath(handle_path)
            painter.setPen(QPen(QColor(64, 200, 224, int(200 * reveal)), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(handle_path)
            # Arrows ◂ ▸ inside handle
            painter.setPen(QColor(64, 200, 224, int(230 * reveal)))
            painter.setFont(QFont("Arial", 11))
            painter.drawText(hx + 2, hy + handle_h // 2 + 5, "◂")
            painter.drawText(hx + handle_w - 13, hy + handle_h // 2 + 5, "▸")

            # --- Label badges ---
            def _draw_badge(text, cx, cy, align_right=False):
                painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(text)
                pad_x, pad_y = 8, 4
                bw, bh = tw + pad_x * 2, fm.height() + pad_y * 2
                bx = (cx - bw) if align_right else cx
                by = cy
                badge = QPainterPath()
                badge.addRoundedRect(float(bx), float(by), float(bw), float(bh), 4, 4)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, int(150 * reveal))))
                painter.drawPath(badge)
                painter.setPen(QColor(255, 255, 255, int(230 * reveal)))
                painter.drawText(bx + pad_x, by + pad_y + fm.ascent(), text)

            if split_x > 80:
                _draw_badge("BEFORE", ox + 8, oy + 8)
            if W - split_x > 80:
                _draw_badge("AFTER", split_x + 8, oy + 8)

        else:
            # Before compare mode activates, show the processed canvas building up
            # tile-by-tile if available; otherwise show the original image.
            painter.drawPixmap(ox, oy, dw, dh,
                               self._after_pm if self._has_after_updates else self._before_pm)
            if self._has_after_updates:
                painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                txt = "Rendering preview"
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(txt)
                bx, by = 10, 10
                bw, bh = tw + 16, fm.height() + 8
                badge = QPainterPath()
                badge.addRoundedRect(float(bx), float(by), float(bw), float(bh), 8, 8)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(7, 13, 20, 210)))
                painter.drawPath(badge)
                painter.setPen(QColor(122, 231, 255, 220))
                painter.drawText(bx + 8, by + 4 + fm.ascent(), txt)

        if self._selection_visible and self._draw_rect is not None:
            x0, y0, sw, sh = self.get_selection_rect()
            rect = self._image_rect_to_screen_rect(float(x0), float(y0), float(sw), float(sh))
            if rect is None:
                self._selection_screen_rect = None
                self._preview_btn.hide()
                painter.end()
                return
            rx, ry, rw, rh = rect
            rw = max(24.0, rw)
            rh = max(24.0, rh)
            self._selection_screen_rect = (rx, ry, rw, rh)
            painter.setPen(QPen(QColor(104, 224, 255, 230), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(float(rx), float(ry), float(rw), float(rh), 8, 8)
            self._update_preview_button_position()
        else:
            self._selection_screen_rect = None
            self._preview_btn.hide()

        # Zoom indicator (top-right, always visible when zoomed)
        if abs(self._zoom - 1.0) > 0.01:
            painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            txt = f"{self._zoom:.0f}x" if self._zoom >= 1.95 else f"{self._zoom:.1f}x"
            fm  = painter.fontMetrics()
            tw  = fm.horizontalAdvance(txt)
            bx, by = W - tw - 18, 6
            bw, bh = tw + 12, fm.height() + 6
            zbg = QPainterPath()
            zbg.addRoundedRect(float(bx), float(by), float(bw), float(bh), 3, 3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 140)))
            painter.drawPath(zbg)
            painter.setPen(QColor(64, 200, 224, 230))
            painter.drawText(bx + 6, by + 3 + fm.ascent(), txt)

        # Minimap (bottom-left) when zoomed in
        if self._zoom > 1.05 and pm_w > 0 and pm_h > 0:
            mm_max = 120
            aspect = pm_w / pm_h
            if aspect >= 1.0:
                mm_w = mm_max
                mm_h = int(mm_max / aspect)
            else:
                mm_h = mm_max
                mm_w = int(mm_max * aspect)
            mm_x, mm_y = 6, H - mm_h - 6
            # Store minimap rect for hit-testing in mouse events
            self._mm_rect = (mm_x, mm_y, mm_w, mm_h)
            # Draw thumbnail background
            painter.setOpacity(0.55)
            painter.drawPixmap(mm_x, mm_y, mm_w, mm_h, self._after_pm)
            painter.setOpacity(1.0)
            # Viewport rectangle inside the minimap
            vp_l = max(0.0, (-ox) / dw) if dw else 0.0
            vp_t = max(0.0, (-oy) / dh) if dh else 0.0
            vp_r = min(1.0, (W - ox) / dw) if dw else 1.0
            vp_b = min(1.0, (H - oy) / dh) if dh else 1.0
            rx = int(mm_x + vp_l * mm_w)
            ry = int(mm_y + vp_t * mm_h)
            rw = max(2, int((vp_r - vp_l) * mm_w))
            rh = max(2, int((vp_b - vp_t) * mm_h))
            painter.setPen(QPen(QColor(64, 200, 224, 200), 1))
            painter.setBrush(QBrush(QColor(64, 200, 224, 30)))
            painter.drawRect(rx, ry, rw, rh)
            # Minimap border
            painter.setPen(QPen(QColor(64, 200, 224, 80), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(mm_x, mm_y, mm_w, mm_h)
        else:
            self._mm_rect = None

        painter.end()


# ============================================================================
# Deep-model purchase dialog  (mirrors DeepPurchaseDialog from inference file)
# ============================================================================

class DeepPurchaseDialog(QDialog):
    """
    Shown when Prism Deep is selected but prism_deep.pt is not found.
    Lets the user buy the model or browse for an already-downloaded file
    and install it automatically into the engine directory.
    """
    def __init__(self, engine_dir: Path, parent=None):
        super().__init__(parent)
        self.engine_dir = engine_dir
        self.setWindowTitle("Prism Deep — Model Not Found")
        self.setModal(True)
        self.setMinimumWidth(500)
        lay = QVBoxLayout(self)

        info = QLabel(
            "<b>The Prism Deep model (prism_deep.pt) was not found.</b><br><br>"
            "To use Prism Deep:<br>"
            "  1. Purchase it from the SyQon website (choose the Siril option)<br>"
            "  2. Once downloaded, click <b>Browse for Downloaded File…</b> and<br>"
            "     select <b>prism_deep.pt</b> — it will be installed automatically."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(info)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        btns   = QHBoxLayout()
        buy    = QPushButton("Buy / Download…")
        buy.setToolTip("Opens the SyQon website to purchase Prism Deep.")
        buy.clicked.connect(self._open_purchase)
        btns.addWidget(buy)

        browse = QPushButton("Browse for Downloaded File…")
        browse.setToolTip("Select the prism_deep.pt file you downloaded.")
        browse.clicked.connect(self._browse_and_install)
        btns.addWidget(browse)

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)

        lay.addLayout(btns)

    def _open_purchase(self):
        webbrowser.open(PRISM_DEEP_URL)

    def _browse_and_install(self):
        from PySide6.QtWidgets import QFileDialog
        src, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prism Deep model file",
            "",
            "PyTorch model (*.pt);;All Files (*)",
        )
        if not src:
            return

        src_path = Path(src)
        if not src_path.exists():
            QMessageBox.warning(self, "SyQon Prism", "Selected file does not exist.")
            return

        if src_path.name != "prism_deep.pt":
            QMessageBox.warning(
                self, "SyQon Prism",
                f"Incorrect file selected.\n\n"
                f"Expected filename:  prism_deep.pt\n"
                f"Selected filename:  {src_path.name}\n\n"
                "Please select the correct file."
            )
            return

        try:
            self.engine_dir.mkdir(parents=True, exist_ok=True)
            dst = self.engine_dir / "prism_deep.pt"
            if dst.resolve() != src_path.resolve():
                import shutil
                self.lbl_status.setText("Installing… please wait.")
                QApplication.processEvents()
                shutil.copy2(str(src_path), str(dst))
            self.lbl_status.setText(f"✓ Installed to: {dst}")
            QMessageBox.information(
                self, "SyQon Prism",
                f"Prism Deep model installed successfully.\n\nLocation: {dst}"
            )
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "SyQon Prism", f"Failed to install model:\n{exc}")


# ============================================================================
# Main GUI window
# ============================================================================

class PrismGUI(QMainWindow):
    """
    Official SyQon Prism GUI for Siril.

    Single-image: shows a before/after split preview updated live per tile.
    Sequence: shows per-frame progress (no live preview).
    """

    def __init__(self, siril, config: dict,
                 raw_data: np.ndarray,
                 before_rgb: np.ndarray,
                 orig_was_mono: bool,
                 orig_dtype: np.dtype,
                 scale: float,
                 engine_dir: Path,
                 is_sequence: bool = False,
                 seq_length: int   = 0):
        super().__init__()
        self.setWindowTitle(f"SyQon Prism — Official Siril Edition v{SCRIPT_VERSION}")
        self.siril         = siril
        self.config        = dict(config)
        self.raw_data      = raw_data      # original Siril pixel data — passed to worker
        self.before_rgb    = before_rgb    # HxWx3 float32 [0,1] — for preview widget only
        self.orig_was_mono = orig_was_mono
        self.orig_dtype    = orig_dtype
        self.scale         = scale
        self.engine_dir    = engine_dir
        self.is_sequence   = is_sequence
        self.seq_length    = seq_length

        self._worker:        Optional[PrismWorker] = None
        self._worker_thread: Optional[QThread]     = None
        self._result:        Optional[np.ndarray]  = None

        self._build_ui()
        self._load_config_into_ui()

    # ------------------------------------------------------------------ UI

    _DARK_STYLESHEET = """
        QMainWindow, QWidget {
            background-color: #0a0d12;
            color: #e9eef5;
            font-family: 'Avenir Next', 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif;
            font-size: 10pt;
        }
        QLabel {
            color: #cbd4df;
            background: transparent;
        }
        QWidget#heroCard, QWidget#previewShell, QWidget#statusCard {
            background-color: #11161d;
            border: 1px solid #202833;
            border-radius: 18px;
        }
        QWidget#controlsShell {
            background-color: transparent;
            border: none;
        }
        QWidget#previewShell {
            background-color: #0f141b;
            border: 1px solid #24303c;
        }
        QLabel#eyebrow {
            color: #7c8a9d;
            font-size: 8pt;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        QLabel#heroTitle {
            color: #f6fbff;
            font-size: 17pt;
            font-weight: 700;
        }
        QLabel#heroSubtitle {
            color: #97a5b8;
            font-size: 9pt;
        }
        QLabel#sectionHint {
            color: #7f8ea2;
            font-size: 8pt;
        }
        QLabel#modelHint {
            color: #90a0b2;
            font-size: 8pt;
        }
        QGroupBox {
            background-color: #11161d;
            border: 1px solid #202833;
            border-radius: 16px;
            margin-top: 8px;
            padding: 18px 14px 14px 14px;
            font-weight: 700;
            color: #eef5fc;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            padding: 4px 10px;
            color: #8adcf0;
            background-color: #0f141b;
            border: 1px solid #202833;
            border-radius: 10px;
        }
        QComboBox {
            background-color: #161d26;
            border: 1px solid #293342;
            border-radius: 10px;
            padding: 6px 34px 6px 10px;
            color: #e9eef5;
            min-height: 26px;
        }
        QComboBox:hover { border-color: #51d0ec; }
        QComboBox:focus { border-color: #7be7ff; }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 28px;
            border-left: 1px solid #293342;
            background-color: #1a2330;
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }
        QComboBox::down-arrow {
            image: none;
            width: 12px;
            height: 12px;
            margin-right: 9px;
        }
        QComboBox QAbstractItemView {
            background-color: #161d26;
            color: #e9eef5;
            selection-background-color: #1f8fb3;
            selection-color: #ffffff;
            border: 1px solid #293342;
            outline: 0;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #161d26;
            border: 1px solid #293342;
            border-radius: 10px;
            padding: 5px 28px 5px 8px;
            color: #e9eef5;
            min-height: 24px;
        }
        QSpinBox:hover, QDoubleSpinBox:hover { border-color: #51d0ec; }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7be7ff; }
        QCheckBox { spacing: 8px; color: #d5dce6; }
        QCheckBox::indicator {
            width: 16px; height: 16px;
            border-radius: 5px;
            border: 1px solid #3a4555;
            background-color: #161d26;
        }
        QCheckBox::indicator:checked {
            background-color: #51d0ec;
            border-color: #51d0ec;
        }
        QSlider::groove:horizontal {
            background: #202833; height: 8px; border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #effaff;
            border: 2px solid #51d0ec;
            width: 18px; height: 18px;
            margin: -7px 0; border-radius: 9px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #157ea0, stop:1 #51d0ec);
            border-radius: 4px;
        }
        QProgressBar {
            background-color: #121923;
            border: 1px solid #1f2934;
            border-radius: 10px;
            text-align: center;
            color: #dfe7f1;
            font-size: 8pt;
            min-height: 14px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #157ea0, stop:1 #51d0ec);
            border-radius: 9px;
        }
        QPushButton {
            background-color: #171f29;
            border: 1px solid #2d3949;
            border-radius: 11px;
            padding: 7px 12px;
            color: #edf3fa;
            font-weight: 600;
            min-height: 20px;
        }
        QPushButton:hover { background-color: #1c2632; border-color: #51d0ec; }
        QPushButton:pressed { background-color: #51d0ec; color: #071019; }
        QPushButton:disabled { background-color: #10151b; color: #536171; border-color: #1b232d; }
        QPushButton#primaryBtn {
            background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #16748e, stop:0.55 #249eb6, stop:1 #6ce3ff);
            color: #041018;
            font-weight: 700;
            font-size: 10pt;
            padding: 9px 18px;
            border: none;
            border-radius: 13px;
        }
        QPushButton#primaryBtn:hover {
            background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #1d8ba7, stop:0.55 #34b9d2, stop:1 #8cecff);
        }
        QPushButton#primaryBtn:pressed { background-color: #16748e; }
        QPushButton#primaryBtn:disabled { background-color: #31404d; color: #7c8a96; }
        QPushButton#cancelBtn {
            background-color: transparent;
            border: 1px solid #3a4656;
            color: #a2afbf;
        }
        QPushButton#cancelBtn:hover { border-color: #ff7272; color: #ff8484; }
        QPushButton#zoomBtn {
            background-color: #161d26;
            border: 1px solid #2d3949;
            border-radius: 10px;
            padding: 3px 8px;
            font-size: 9pt;
            min-height: 18px; min-width: 24px;
            color: #dbe3ed;
        }
        QPushButton#zoomBtn:hover { border-color: #51d0ec; color: #8fe8fa; }
    """

    def _build_ui(self):
        self.setStyleSheet(self._DARK_STYLESHEET)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 20, 22, 20)
        root.setSpacing(12)

        # Header
        hero = QFrame()
        hero.setObjectName("heroCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(18, 14, 18, 14)
        hero_layout.setSpacing(12)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(3)
        title = QLabel("SyQon Prism")
        title.setObjectName("heroTitle")
        title_wrap.addWidget(title)

        subtitle = QLabel("Official Siril Edition for denoising.")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        title_wrap.addWidget(subtitle)
        hero_layout.addLayout(title_wrap, stretch=1)

        root.addWidget(hero)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)
        root.addLayout(content_row, stretch=1)

        controls_shell = QFrame()
        controls_shell.setObjectName("controlsShell")
        controls_layout = QVBoxLayout(controls_shell)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_shell.setFixedWidth(430)
        content_row.addWidget(controls_shell, stretch=0)

        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        content_row.addLayout(right_column, stretch=1)

        # Preview (single image only)
        if not self.is_sequence:
            preview_shell = QFrame()
            preview_shell.setObjectName("previewShell")
            preview_shell_layout = QVBoxLayout(preview_shell)
            preview_shell_layout.setContentsMargins(14, 12, 14, 12)
            preview_shell_layout.setSpacing(8)

            preview_meta = QHBoxLayout()
            preview_meta.setSpacing(8)
            preview_title = QLabel("Preview")
            preview_title.setStyleSheet("color: #f3f7fc; font-size: 11pt; font-weight: 700;")
            preview_meta.addWidget(preview_title)
            preview_meta.addStretch()
            zoom_label = QLabel("Navigation")
            zoom_label.setStyleSheet("color: #8a98aa; font-size: 8pt; font-weight: 600;")
            preview_meta.addWidget(zoom_label)
            self.btn_zoom_out = QPushButton("−")
            self.btn_zoom_out.setObjectName("zoomBtn")
            self.btn_zoom_out.setFixedSize(28, 24)
            self.btn_zoom_out.setToolTip("Zoom out")
            preview_meta.addWidget(self.btn_zoom_out)

            self.lbl_zoom = QLabel("1.0x")
            self.lbl_zoom.setStyleSheet("color: #888; font-size: 8pt; font-family: monospace;")
            self.lbl_zoom.setFixedWidth(42)
            self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_meta.addWidget(self.lbl_zoom)

            self.btn_zoom_in = QPushButton("+")
            self.btn_zoom_in.setObjectName("zoomBtn")
            self.btn_zoom_in.setFixedSize(28, 24)
            self.btn_zoom_in.setToolTip("Zoom in")
            preview_meta.addWidget(self.btn_zoom_in)

            self.btn_zoom_fit = QPushButton("Fit")
            self.btn_zoom_fit.setObjectName("zoomBtn")
            self.btn_zoom_fit.setFixedSize(36, 24)
            self.btn_zoom_fit.setToolTip("Reset zoom to fit")
            preview_meta.addWidget(self.btn_zoom_fit)
            preview_shell_layout.addLayout(preview_meta)

            self.preview = SplitPreviewWidget(self.before_rgb, self)
            self.preview.setMinimumHeight(300)
            self.preview.setStyleSheet(
                "border: 1px solid #26313c; border-radius: 14px; "
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                "stop:0 #0b1016, stop:1 #151d27);"
            )
            preview_shell_layout.addWidget(self.preview, stretch=3)
            self.btn_zoom_out.clicked.connect(self.preview.zoom_out)
            self.btn_zoom_in.clicked.connect(self.preview.zoom_in)
            self.btn_zoom_fit.clicked.connect(self.preview.reset_zoom)

            self.lbl_preview_disabled = QLabel(
                "<center><br><br>"
                "<span style='font-size: 11pt; color:#d9e3ef; font-weight:600;'>Live preview disabled</span><br><br>"
                "<span style='color:#8fa1b4;'>Processing still runs normally.<br>"
                "Enable <b>live tile preview</b> to watch the render build up and compare the final result.</span>"
                "</center>"
            )
            self.lbl_preview_disabled.setMinimumHeight(300)
            self.lbl_preview_disabled.setVisible(False)
            self.lbl_preview_disabled.setStyleSheet(
                "border: 1px dashed #2a3644; border-radius: 14px; background: #0d131a; padding: 24px;"
            )
            preview_shell_layout.addWidget(self.lbl_preview_disabled, stretch=3)

            self.lbl_preview_hint = QLabel(
                "Tiles render live. The before/after divider fades in when the full pass is complete."
            )
            self.lbl_preview_hint.setObjectName("sectionHint")
            preview_shell_layout.addWidget(self.lbl_preview_hint)

            right_column.addWidget(preview_shell, stretch=1)

            self.preview.zoom_changed.connect(self._on_zoom_changed)
        else:
            self.preview               = None
            self.lbl_preview_disabled  = None
            self.lbl_preview_hint      = None
            self.btn_zoom_in           = None
            self.btn_zoom_out          = None
            self.btn_zoom_fit          = None
            self.lbl_zoom              = None
            seq_lbl = QLabel(
                f"<b>Sequence mode</b> — {self.seq_length} included frames. "
                "Live preview unavailable in sequence mode."
            )
            seq_lbl.setStyleSheet(
                "color: #8b99ab; padding: 18px; background: #11161d; "
                "border: 1px solid #202833; border-radius: 18px;"
            )
            right_column.addWidget(seq_lbl, stretch=1)

        # ---- Settings ----
        essentials_box = QGroupBox("Essentials")
        essentials_form = QVBoxLayout(essentials_box)
        essentials_form.setContentsMargins(14, 20, 14, 14)
        essentials_form.setSpacing(10)

        lbl_model = QLabel("Model")
        lbl_model.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
        essentials_form.addWidget(lbl_model)
        model_select_row = QHBoxLayout()
        model_select_row.setSpacing(8)
        self.cmb_model = QComboBox()
        self.cmb_model.addItem("Prism Mini", "mini")
        self.cmb_model.addItem("Prism Deep", "deep")
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self._cmb_model_chevron = QLabel(">", essentials_box)
        self._cmb_model_chevron_parent = essentials_box
        self._cmb_model_chevron.setStyleSheet("color: #9bcddd; font-size: 14pt; font-weight: 700;")
        self._cmb_model_chevron.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cmb_model_chevron.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.cmb_model.installEventFilter(self)
        model_select_row.addWidget(self.cmb_model, 1)
        essentials_form.addLayout(model_select_row)

        self.lbl_model_hint = QLabel("")
        self.lbl_model_hint.setObjectName("modelHint")
        self.lbl_model_hint.setWordWrap(True)
        essentials_form.addWidget(self.lbl_model_hint)

        strength_row = QHBoxLayout()
        strength_row.setSpacing(8)
        lbl_strength = QLabel("Strength")
        lbl_strength.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
        strength_row.addWidget(lbl_strength)
        strength_row.addStretch()
        self.sld_mod = QSlider(Qt.Orientation.Horizontal)
        self.sld_mod.setRange(0, 100)
        self.lbl_mod = QLabel("100%")
        self.lbl_mod.setFixedWidth(48)
        self.lbl_mod.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_mod.setStyleSheet("color: #f2f8ff; font-size: 9pt; font-weight: 700;")
        self.sld_mod.valueChanged.connect(lambda v: self.lbl_mod.setText(f"{v}%"))
        strength_row.addWidget(self.lbl_mod)
        essentials_form.addLayout(strength_row)
        essentials_form.addWidget(self.sld_mod)

        self.chk_live_preview = None
        if not self.is_sequence:
            preview_opts_label = QLabel("Preview")
            preview_opts_label.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
            essentials_form.addWidget(preview_opts_label)

            self.chk_live_preview = QCheckBox("Live tile preview")
            self.chk_live_preview.setChecked(True)
            self.chk_live_preview.setToolTip(
                "Updates the preview while inference is running.\n"
                "The final before/after compare appears when the tiled pass completes."
            )
            self.chk_live_preview.toggled.connect(self._on_live_preview_toggled)
            essentials_form.addWidget(self.chk_live_preview)

            self.chk_show_stretch = QCheckBox("Show IHS stretch in preview")
            self.chk_show_stretch.setChecked(True)
            self.chk_show_stretch.setToolTip(
                "Controls the preview display only.\n"
                "Before uses the temporary stretch when enabled so faint detail is easier to inspect."
            )
            essentials_form.addWidget(self.chk_show_stretch)
        else:
            self.chk_show_stretch = None
            self.spin_ihs_target = None

        controls_layout.addWidget(essentials_box)

        advanced_box = QGroupBox("Advanced")
        advanced_form = QVBoxLayout(advanced_box)
        advanced_form.setContentsMargins(14, 20, 14, 14)
        advanced_form.setSpacing(10)

        top_adv_row = QHBoxLayout()
        top_adv_row.setSpacing(10)
        tile_col = QVBoxLayout()
        tile_col.setSpacing(4)
        tile_lbl = QLabel("Tile size")
        tile_lbl.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
        tile_col.addWidget(tile_lbl)
        self.spin_tile = QSpinBox()
        self.spin_tile.setRange(64, 2048); self.spin_tile.setSingleStep(64)
        tile_col.addWidget(self.spin_tile)
        top_adv_row.addLayout(tile_col, 1)

        overlap_col = QVBoxLayout()
        overlap_col.setSpacing(4)
        overlap_lbl = QLabel("Overlap")
        overlap_lbl.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
        overlap_col.addWidget(overlap_lbl)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(8, 512); self.spin_overlap.setSingleStep(8)
        overlap_col.addWidget(self.spin_overlap)
        top_adv_row.addLayout(overlap_col, 1)
        advanced_form.addLayout(top_adv_row)

        second_adv_row = QHBoxLayout()
        second_adv_row.setSpacing(10)
        pad_col = QVBoxLayout()
        pad_col.setSpacing(4)
        pad_lbl = QLabel("Edge pad")
        pad_lbl.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
        pad_col.addWidget(pad_lbl)
        self.spin_pad = QSpinBox()
        self.spin_pad.setRange(0, 1024); self.spin_pad.setSingleStep(16)
        self.spin_pad.setToolTip(
            "Reflect-pads image before tiling to prevent border artefacts.\n"
            "Recommended: >= overlap. Set 0 to disable."
        )
        pad_col.addWidget(self.spin_pad)
        pad_hint = QLabel("Recommended: >= overlap")
        pad_hint.setObjectName("sectionHint")
        pad_col.addWidget(pad_hint)
        second_adv_row.addLayout(pad_col, 1)

        if not self.is_sequence:
            from PySide6.QtWidgets import QDoubleSpinBox
            ihs_col = QVBoxLayout()
            ihs_col.setSpacing(4)
            ihs_lbl = QLabel("Target median")
            ihs_lbl.setStyleSheet("color: #dfe7f1; font-size: 9pt; font-weight: 600;")
            ihs_col.addWidget(ihs_lbl)
            self.spin_ihs_target = QDoubleSpinBox()
            self.spin_ihs_target.setRange(0.05, 0.50)
            self.spin_ihs_target.setSingleStep(0.01)
            self.spin_ihs_target.setDecimals(3)
            self.spin_ihs_target.setValue(_DEFAULT_TARGET)
            self.spin_ihs_target.setToolTip(
                "Target median brightness for the temporary IHS stretch.\n"
                "Lower values = more aggressive stretch.\n"
                "Higher values = gentler stretch. Default: %.2f" % _DEFAULT_TARGET
            )
            ihs_col.addWidget(self.spin_ihs_target)
            ihs_hint = QLabel("Preview-only control")
            ihs_hint.setObjectName("sectionHint")
            ihs_col.addWidget(ihs_hint)
            stretch_preset_row = QHBoxLayout()
            stretch_preset_row.setSpacing(6)
            self.btn_stretch_soft = QPushButton("Soft")
            self.btn_stretch_soft.setObjectName("zoomBtn")
            self.btn_stretch_soft.clicked.connect(lambda: self._apply_stretch_preset(0.20))
            stretch_preset_row.addWidget(self.btn_stretch_soft)
            self.btn_stretch_std = QPushButton("Std")
            self.btn_stretch_std.setObjectName("zoomBtn")
            self.btn_stretch_std.clicked.connect(lambda: self._apply_stretch_preset(0.15))
            stretch_preset_row.addWidget(self.btn_stretch_std)
            self.btn_stretch_boost = QPushButton("Boost")
            self.btn_stretch_boost.setObjectName("zoomBtn")
            self.btn_stretch_boost.clicked.connect(lambda: self._apply_stretch_preset(0.11))
            stretch_preset_row.addWidget(self.btn_stretch_boost)
            stretch_preset_row.addStretch()
            ihs_col.addLayout(stretch_preset_row)
            second_adv_row.addLayout(ihs_col, 1)
        advanced_form.addLayout(second_adv_row)

        self.chk_amp = QCheckBox("AMP mixed precision (CUDA / MPS)")
        advanced_form.addWidget(self.chk_amp)

        self.chk_no_gpu = QCheckBox("Force CPU (disable GPU)")
        self.chk_no_gpu.stateChanged.connect(self._on_gpu_toggle)
        advanced_form.addWidget(self.chk_no_gpu)

        if not self.is_sequence:
            self.chk_show_stretch.toggled.connect(self._on_stretch_toggled)
            self.spin_ihs_target.valueChanged.connect(self._on_stretch_toggled)

        controls_layout.addWidget(advanced_box)
        controls_layout.addStretch()

        # Progress
        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(14, 10, 14, 10)
        status_layout.setSpacing(6)

        status_title = QLabel("Process")
        status_title.setStyleSheet("color: #f3f7fc; font-size: 11pt; font-weight: 700;")
        status_layout.addWidget(status_title)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setTextVisible(True)
        self.pbar.setFormat("%p%")
        status_layout.addWidget(self.pbar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #91a0b1; font-size: 9pt; padding: 2px 2px;")
        status_layout.addWidget(self.lbl_status)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._on_preview_button)
        self.btn_run = QPushButton("▶  Denoise")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setDefault(True)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_close = QPushButton("Close")
        self.btn_preview.setMinimumWidth(92)
        self.btn_run.setMinimumWidth(110)
        self.btn_cancel.setMinimumWidth(84)
        self.btn_close.setMinimumWidth(84)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_close)
        status_layout.addLayout(btn_row)
        right_column.addWidget(status_card, stretch=0)

        if self.preview is not None:
            self.preview.preview_requested.connect(self._on_preview_tile)
        self._sync_model_ui()
        self._position_model_chevron()
        self._set_stage("pending")
        self.setMinimumSize(1120, 640)

    # ---------------------------------------------------------------- config

    def _load_config_into_ui(self):
        cfg = self.config
        model_str = cfg.get("model", "mini")
        if "Deep" in model_str or model_str == "deep":
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(0)
        self.spin_tile.setValue(int(cfg.get("tile_size", 512)))
        self.spin_overlap.setValue(int(cfg.get("overlap", 64)))
        self.spin_pad.setValue(int(cfg.get("pad", 96)))
        self.sld_mod.setValue(int(cfg.get("modulation", 100)))
        self.chk_amp.setChecked(bool(cfg.get("use_amp", False)))
        if self.spin_ihs_target is not None:
            self.spin_ihs_target.setValue(float(cfg.get("ihs_target", _DEFAULT_TARGET)))
        self._sync_model_ui()
        # Apply stretch to preview immediately if checkbox is on
        self._on_stretch_toggled()

    def _gather_config(self) -> dict:
        return {
            "model":      self.cmb_model.currentData(),
            "tile_size":  self.spin_tile.value(),
            "overlap":    self.spin_overlap.value(),
            "pad":        self.spin_pad.value(),
            "modulation": self.sld_mod.value(),
            "use_amp":    self.chk_amp.isChecked(),
            "ihs_target": float(self.spin_ihs_target.value())
                          if self.spin_ihs_target is not None
                          else _DEFAULT_TARGET,
        }

    def eventFilter(self, obj, event):
        if obj is self.cmb_model and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.Move,
        ):
            self._position_model_chevron()
        return super().eventFilter(obj, event)

    def _position_model_chevron(self):
        if not hasattr(self, "_cmb_model_chevron") or self._cmb_model_chevron is None:
            return
        h = self.cmb_model.height()
        if h <= 0:
            return
        parent = getattr(self, "_cmb_model_chevron_parent", self.cmb_model)
        pos = self.cmb_model.mapTo(parent, self.cmb_model.rect().topRight())
        self._cmb_model_chevron.resize(16, h)
        self._cmb_model_chevron.move(max(0, pos.x() - 24), pos.y())
        self._cmb_model_chevron.raise_()
        self._cmb_model_chevron.show()

    # ---------------------------------------------------------------- model

    def _sync_model_ui(self):
        current = self.cmb_model.currentData()
        is_deep = current == "deep"
        if is_deep:
            if _deep_path().exists():
                self.lbl_model_hint.setText(
                    "Prism Deep is installed and ready. Expect slower passes, but more denoise headroom."
                )
            else:
                self.lbl_model_hint.setText(
                    "Prism Deep is selected but not installed yet. Select it to import the downloaded model file when prompted."
                )
        else:
            self.lbl_model_hint.setText(
                "Prism Mini downloads automatically on first run and is the fastest starting point for most images."
            )

    def _set_stage(self, stage: str):
        return

    def _apply_stretch_preset(self, value: float):
        if self.spin_ihs_target is None:
            return
        self.spin_ihs_target.setValue(float(value))
        self.lbl_status.setText(f"Stretch preset set to {value:.2f}.")

    def _on_model_changed(self):
        self._sync_model_ui()
        if self.cmb_model.currentData() == "deep":
            if not _deep_path().exists():
                dlg = DeepPurchaseDialog(self.engine_dir, self)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    # revert to mini
                    self.cmb_model.setCurrentIndex(0)
                    self._sync_model_ui()

    def _current_ckpt_path(self) -> Optional[Path]:
        if self.cmb_model.currentData() == "deep":
            p = _deep_path()
            if not p.exists():
                QMessageBox.warning(self, "SyQon Prism",
                    "Prism Deep model not found.\nPlease install it first.")
                return None
            return p
        return _mini_path()

    # ---------------------------------------------------------------- GPU toggle

    def _on_gpu_toggle(self, _state):
        global _MODEL, _DEVICE, _AMP_ENABLED, _RESIDUAL_MODE
        _MODEL = None   # force reload on correct device next run

    # ---------------------------------------------------------------- stretch preview

    def _on_live_preview_toggled(self, enabled: bool):
        """Show the split-preview widget or the disabled placeholder."""
        if self.is_sequence or self.preview is None:
            return
        self.preview.setVisible(enabled)
        self.lbl_preview_disabled.setVisible(not enabled)
        self.lbl_preview_hint.setVisible(enabled)
        # Show/hide zoom controls together with the preview
        for w in (self.btn_zoom_in, self.btn_zoom_out,
                  self.btn_zoom_fit, self.lbl_zoom):
            if w is not None:
                w.setVisible(enabled)

    def _on_zoom_changed(self, zoom: float):
        """Update the zoom label when the preview zoom level changes."""
        if self.lbl_zoom is not None:
            txt = f"{zoom:.0f}x" if zoom >= 1.95 else f"{zoom:.1f}x"
            self.lbl_zoom.setText(txt)



    def _on_stretch_toggled(self, *_):
        """Immediately update the preview when stretch checkbox or target changes."""
        if self.is_sequence or self._worker_thread is not None:
            return  # don't interfere during inference
        if self.chk_show_stretch is None:
            return
        if self.chk_show_stretch.isChecked():
            target = float(self.spin_ihs_target.value())
            params = compute_ihs_params(self.before_rgb, target=target)
            stretched = apply_ihs(self.before_rgb, params)
            self.preview.reset_both(np.clip(stretched, 0.0, 1.0))
        else:
            self.preview.reset_both(self.before_rgb)

    def _on_preview_button(self):
        if self.is_sequence or self.preview is None or self._worker_thread is not None:
            return
        if not self.preview._selection_visible:
            self.preview.reset_selection(512)
            self.preview.set_selection_visible(True)
            self.lbl_status.setText("Move the 512 px square, then click Run Preview.")
            return
        self._on_preview_tile()

    def _on_preview_tile(self):
        """Run a small optional center-crop preview without touching the Siril image."""
        if self.is_sequence or self.preview is None or self._worker_thread is not None:
            return

        ckpt = self._current_ckpt_path()
        if ckpt is None:
            return

        cfg = self._gather_config()
        x0, y0, sw, sh = self.preview.get_selection_rect()
        x1 = x0 + sw
        y1 = y0 + sh
        crop = self.before_rgb[y0:y1, x0:x1, :].astype(np.float32, copy=True)

        if crop.size == 0:
            return

        target = float(cfg.get("ihs_target", _DEFAULT_TARGET))
        ihs_params = compute_ihs_params(crop, target=target)
        full_before = self.before_rgb.astype(np.float32, copy=True)
        before_display = (
            apply_ihs(crop, ihs_params)
            if self.chk_show_stretch is not None and self.chk_show_stretch.isChecked()
            else crop
        )

        if self.chk_show_stretch is not None and self.chk_show_stretch.isChecked():
            full_before_display = apply_ihs(full_before, compute_ihs_params(full_before, target=target))
        else:
            full_before_display = full_before

        self.preview.reset_both(np.clip(full_before_display, 0.0, 1.0))
        self.lbl_status.setText(f"Rendering test tile… {crop.shape[1]}×{crop.shape[0]}")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        QApplication.processEvents()

        try:
            tile_for_preview = min(int(cfg["tile_size"]), min(crop.shape[0], crop.shape[1]))
            overlap_for_preview = min(int(cfg["overlap"]), max(0, tile_for_preview - 1))
            result = process_image(
                crop,
                tile=tile_for_preview,
                overlap=overlap_for_preview,
                use_amp=bool(cfg["use_amp"]),
                use_gpu=not self.chk_no_gpu.isChecked(),
                modulation=cfg["modulation"] / 100.0,
                model_path=ckpt,
                ihs_target=target,
            )
            result = np.asarray(result, dtype=np.float32)
            if result.ndim == 2:
                result = np.stack([result] * 3, axis=-1)
            elif result.ndim == 3 and result.shape[2] == 1:
                result = np.repeat(result, 3, axis=2)
            after_display = apply_ihs(result[..., :3], ihs_params)
            self.preview.update_tile(y0, x0, crop.shape[0], crop.shape[1],
                                     np.clip(after_display, 0.0, 1.0))
            self.preview.focus_selection()
            self.lbl_status.setText(
                f"Test tile preview ready ({crop.shape[1]}×{crop.shape[0]} selection)."
            )
        except Exception as exc:
            self.preview.reset_both(self.before_rgb)
            QMessageBox.critical(self, "SyQon Prism",
                                 f"Tile preview error:\n{str(exc)[:800]}")
            self.lbl_status.setText("Tile preview failed.")
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    # ---------------------------------------------------------------- run / cancel

    def _on_run(self):
        if self._worker_thread and self._worker_thread.isRunning():
            return

        ckpt = self._current_ckpt_path()
        if ckpt is None:
            return

        cfg = self._gather_config()
        self.config.update(cfg)
        save_config(self.config, self.siril)

        self._set_busy(True)
        self._set_stage("load")
        self.lbl_status.setText("Initialising…")
        self.pbar.setValue(0)
        self._eta_start_time = None
        self._eta_start_pct  = None

        if not self.is_sequence:
            self.preview.set_selection_visible(False)
            self.preview.reset_zoom()
            pad        = int(cfg["pad"])
            ihs_target = float(cfg.get("ihs_target", _DEFAULT_TARGET))

            # Build the padded linear image — this is the true before for
            # the preview so both panels live in exactly the same pixel space.
            padded_linear, _ = _pad_reflect(self.before_rgb, pad)

            # Optionally build the IHS-stretched version of the padded image.
            if (self.chk_show_stretch is not None
                    and self.chk_show_stretch.isChecked()):
                ihs_params_padded = compute_ihs_params(padded_linear, target=ihs_target)
                padded_before_for_preview = apply_ihs(padded_linear, ihs_params_padded)
            else:
                padded_before_for_preview = padded_linear

            # Store both for use in _on_finished_ok
            self._padded_linear  = padded_linear
            self._padded_preview = padded_before_for_preview  # stretched or linear

            # Initialise both panels to the same padded image so tile
            # updates paint into an already-consistent canvas.
            self.preview.reset_both(padded_before_for_preview)

        self._worker = PrismWorker(
            raw           = self.raw_data,
            tile          = cfg["tile_size"],
            overlap       = cfg["overlap"],
            pad           = cfg["pad"],
            use_amp       = cfg["use_amp"],
            use_gpu       = not self.chk_no_gpu.isChecked(),
            modulation    = cfg["modulation"] / 100.0,
            model_path    = ckpt,
            orig_dtype    = self.orig_dtype,
            scale         = self.scale,
            orig_was_mono = self.orig_was_mono,
            ihs_target    = cfg.get("ihs_target", _DEFAULT_TARGET),
        )
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.tile_done.connect(self._on_tile_done)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_err.connect(self._on_finished_err)
        # Use lambda so quit() fires regardless of how many args the signal carries
        self._worker.finished_ok.connect(lambda *_: self._worker_thread.quit())
        self._worker.finished_err.connect(lambda *_: self._worker_thread.quit())
        self._worker_thread.finished.connect(self._cleanup_thread)

        self._worker_thread.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self._set_stage("tile")
            self.lbl_status.setText("Cancelling… finishing current tile.")

    def _on_progress(self, pct: int):
        import time
        self.pbar.setValue(pct)
        if pct < 10:
            self._set_stage("load")
            self.lbl_status.setText("Loading model…")
        elif pct < 15:
            self._set_stage("prepare")
            self.lbl_status.setText("Preparing (IHS stretch)…")
        elif pct < 87:
            self._set_stage("tile")
            # Start ETA tracking once tiling begins (pct ≥ 15)
            if self._eta_start_time is None and pct >= 15:
                self._eta_start_time = time.monotonic()
                self._eta_start_pct  = pct
            eta_str = ""
            if (self._eta_start_time is not None
                    and pct > self._eta_start_pct):
                elapsed = time.monotonic() - self._eta_start_time
                done    = pct - self._eta_start_pct
                remain  = 87 - pct
                if done > 0 and remain > 0:
                    secs = elapsed / done * remain
                    if secs >= 60:
                        eta_str = f" — ~{int(secs // 60)}m {int(secs % 60):02d}s left"
                    else:
                        eta_str = f" — ~{int(secs)}s left"
            self.lbl_status.setText(f"Processing tiles… {pct}%{eta_str}")
        else:
            self._set_stage("finalize")
            self.lbl_status.setText("Inverting stretch / finalising…")

    def _on_tile_done(self, y0: int, x0: int, ph: int, pw: int, patch):
        if self.is_sequence:
            return
        if self.chk_live_preview is None or not self.chk_live_preview.isChecked():
            return
        # Show the hint label when the first tile arrives
        if self.lbl_preview_hint is not None and not self.lbl_preview_hint.isVisible():
            self.lbl_preview_hint.setVisible(True)
        self.preview.update_tile(y0, x0, ph, pw, np.asarray(patch))

    def _on_finished_ok(self, denoised_for_siril, denoised_padded_rgb):
        denoised_for_siril  = np.asarray(denoised_for_siril)
        denoised_padded_rgb = np.asarray(denoised_padded_rgb, dtype=np.float32)
        self._result = denoised_for_siril   # kept for get_result() if needed

        if not self.is_sequence:
            # The after panel is already fully painted by the live tile updates —
            # don't touch it.  Just restore the before panel to its correct state.
            show_stretch = (self.chk_show_stretch is not None
                            and self.chk_show_stretch.isChecked())
            before_preview = self._padded_preview if show_stretch else self._padded_linear
            self.preview.reset_before(before_preview)
            self.preview.activate_split()

        self._set_busy(False)
        self._set_stage("done")
        self.lbl_status.setText("Complete! Pushing to Siril…")

        # Push directly to Siril while the event loop is still running,
        # mirroring how the original community script works.
        try:
            self.siril.undo_save_state("SyQon Prism — denoising")
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(denoised_for_siril)
            print("Denoised image sent to Siril.")
            self.lbl_status.setText("Complete! ✓ Image updated in Siril.")
        except Exception as exc:
            print(f"Warning: could not update Siril image: {exc}")
            self.lbl_status.setText(f"Complete — but Siril push failed: {exc}")

    def _on_finished_err(self, err: str):
        self._set_busy(False)
        if not self.is_sequence:
            # Restore the padded linear before so the panel isn't left
            # stuck on the stretched version.
            before = getattr(self, "_padded_linear", self.before_rgb)
            self.preview.reset_before(before)
        if err == "__cancelled__":
            self._set_stage("cancelled")
            self.lbl_status.setText("Cancelled.")
            return
        self._set_stage("error")
        QMessageBox.critical(self, "SyQon Prism",
                             f"Processing error:\n{err[:800]}")
        self.lbl_status.setText("Error.")

    def _cleanup_thread(self):
        self._worker_thread = None
        self._worker        = None

    def _set_busy(self, busy: bool):
        self.btn_run.setEnabled(not busy)
        self.btn_preview.setEnabled((not busy) and (not self.is_sequence))
        self.btn_cancel.setEnabled(busy)
        self.cmb_model.setEnabled(not busy)
        self.spin_tile.setEnabled(not busy)
        self.spin_overlap.setEnabled(not busy)
        self.spin_pad.setEnabled(not busy)
        self.sld_mod.setEnabled(not busy)
        self.chk_amp.setEnabled(not busy)
        self.chk_no_gpu.setEnabled(not busy)
        if self.preview is not None:
            self.preview.set_preview_button_enabled(not busy)
        if hasattr(self, "btn_stretch_soft"):
            self.btn_stretch_soft.setEnabled(not busy)
            self.btn_stretch_std.setEnabled(not busy)
            self.btn_stretch_boost.setEnabled(not busy)
        # Preview options stay interactive during processing
        # (user can toggle live preview on/off mid-run)

    def get_result(self) -> Optional[np.ndarray]:
        return self._result

    def closeEvent(self, ev):
        if self._worker and self._worker_thread and self._worker_thread.isRunning():
            self._worker.cancel()
            self._worker_thread.wait(8000)
        super().closeEvent(ev)


# ============================================================================
# Sequence headless processing
# ============================================================================

def _run_sequence(siril, config: dict, engine_dir: Path, use_gpu: bool):
    sequence = siril.get_seq()
    included = [i for i in range(sequence.number) if sequence.imgparam[i].incl]
    print(f"Processing {len(included)} included frames…")

    ckpt_path = (_deep_path() if config.get("model") == "deep" else _mini_path())
    if not ckpt_path.exists():
        print(f"ERROR: model not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    cwd = siril.get_siril_wd()

    for idx, frame_num in enumerate(included):
        print(f"\n--- Frame {frame_num} ({idx + 1}/{len(included)}) ---")
        try:
            frame  = siril.get_seq_frame(frame_num)
            xrgb, orig_dtype, scale, orig_was_mono = _prepare_for_inference(frame.data)

            padded, orig_hw = _pad_reflect(xrgb, int(config.get("pad", 96)))

            def _prog(v):
                print(f"  [{v:3d}%]", end="\r")

            denoised_padded = process_image(
                padded,
                tile              = int(config.get("tile_size", 512)),
                overlap           = int(config.get("overlap", 64)),
                use_amp           = bool(config.get("use_amp", False)),
                use_gpu           = use_gpu,
                modulation        = int(config.get("modulation", 100)) / 100.0,
                model_path        = ckpt_path,
                progress_callback = _prog,
            )
            print()

            # Ensure HxWx3 for unpadding
            if denoised_padded.ndim == 2:
                denoised_padded = np.stack([denoised_padded] * 3, axis=-1)
            denoised_hwc3 = _unpad(denoised_padded, orig_hw, int(config.get("pad", 96)))

            # Collapse to mono or produce Siril planar layout (RGB, no channel swap)
            if orig_was_mono:
                result = _restore_dtype(
                    denoised_hwc3.mean(axis=2)[np.newaxis, ...],  # (1,H,W)
                    orig_dtype, scale
                )
            else:
                result = _restore_dtype(
                    denoised_hwc3.transpose(2, 0, 1),             # (3,H,W) RGB
                    orig_dtype, scale
                )

            filename = siril.get_seq_frame_filename(frame_num)
            base     = os.path.splitext(os.path.basename(filename))[0]
            new_fn   = os.path.join(cwd,
                                    ("denoised_" + base) if not base.startswith("denoised_")
                                    else base)
            siril.save_image_file(result, frame.header, new_fn)
            print(f"  Saved: {new_fn}")

        except Exception as exc:
            import traceback
            print(f"Error on frame {frame_num}: {exc}")
            traceback.print_exc()

    try:
        siril.create_new_seq(f"denoised_{sequence.seqname}")
        siril.cmd("load_seq", f"denoised_{sequence.seqname}")
        print(f"\nCreated sequence: denoised_{sequence.seqname}")
    except Exception as exc:
        print(f"Warning: could not create output sequence: {exc}")

    print(f"\nSequence complete — {len(included)} frames processed.")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SyQon Prism — Official Siril Edition"
    )
    parser.add_argument("--tile-size",   type=int,   default=None)
    parser.add_argument("--overlap",     type=int,   default=None)
    parser.add_argument("--pad",         type=int,   default=None)
    parser.add_argument("--modulation",  type=float, default=None,
                        help="0.0 (original) – 1.0 (full denoise)")
    parser.add_argument("--model",       type=str,   default=None,
                        choices=["mini", "deep"])
    parser.add_argument("--no-gpu",      action="store_true")
    parser.add_argument("--force-update",action="store_true")
    args = parser.parse_args()

    # ---- Connect to Siril ----
    siril = None
    try:
        siril = s.SirilInterface()
        siril.connect()
        print("Connected to Siril.")
    except Exception as exc:
        print(f"Could not connect to Siril: {exc}", file=sys.stderr)
        sys.exit(1)

    is_single = siril.is_image_loaded()
    is_seq    = siril.is_sequence_loaded()
    if not is_single and not is_seq:
        print("Error: No image or sequence loaded.", file=sys.stderr)
        sys.exit(1)

    # ---- Resolve engine dir and model paths ----
    engine_dir = _resolve_engine_dir(siril)

    # ---- Update check (mirrors community script cadence) ----
    should_check = _should_check_updates(engine_dir, args.force_update)
    if should_check:
        _touch_stamp(engine_dir)

    # ---- Load / merge config ----
    config = load_config(siril)
    if args.tile_size  is not None: config["tile_size"]   = args.tile_size
    if args.overlap    is not None: config["overlap"]     = args.overlap
    if args.pad        is not None: config["pad"]         = args.pad
    if args.modulation is not None: config["modulation"]  = int(args.modulation * 100)
    if args.model      is not None: config["model"]       = args.model

    use_gpu = not args.no_gpu

    # ---- GUI mode ----
    if not (siril.is_cli() and len(sys.argv) > 1):
        app = QApplication.instance() or QApplication(sys.argv)

        # Ensure Mini model present before opening GUI
        if config.get("model", "mini") != "deep":
            ensure_mini_model(engine_dir, siril, should_check)

        if is_single:
            try:
                image = siril.get_image()
                raw   = image.data
            except Exception as exc:
                QMessageBox.critical(None, "SyQon Prism",
                                     f"Could not load image:\n{exc}")
                sys.exit(1)

            # HxWx3 float32 for the preview widget only
            xrgb, orig_dtype, scale, orig_was_mono = _prepare_for_inference(raw)

            gui = PrismGUI(
                siril         = siril,
                config        = config,
                raw_data      = raw,           # original Siril data for the worker
                before_rgb    = xrgb,          # HxWx3 preview image
                orig_was_mono = orig_was_mono,
                orig_dtype    = orig_dtype,
                scale         = scale,
                engine_dir    = engine_dir,
                is_sequence   = False,
            )
            gui.show()
            app.exec()

            # The Siril push is handled directly inside _on_finished_ok
            # while the event loop is running. Nothing to do here.

        else:   # sequence GUI
            sequence = siril.get_seq()
            included = [i for i in range(sequence.number)
                        if sequence.imgparam[i].incl]

            try:
                first = siril.get_seq_frame(included[0])
                raw_rep = first.data
                xrgb_rep, orig_dtype_rep, scale_rep, mono_rep = \
                    _prepare_for_inference(raw_rep)
            except Exception as exc:
                QMessageBox.critical(None, "SyQon Prism",
                                     f"Could not load representative frame:\n{exc}")
                sys.exit(1)

            gui = PrismGUI(
                siril         = siril,
                config        = config,
                raw_data      = raw_rep,
                before_rgb    = xrgb_rep,
                orig_was_mono = mono_rep,
                orig_dtype    = orig_dtype_rep,
                scale         = scale_rep,
                engine_dir    = engine_dir,
                is_sequence   = True,
                seq_length    = len(included),
            )
            gui.show()
            app.exec()

            # User closed the GUI — re-run headless with final saved config
            final_cfg = gui._gather_config()
            save_config(final_cfg, siril)
            if gui.get_result() is not None:
                _run_sequence(siril, final_cfg, engine_dir, use_gpu)

    # ---- CLI / programmatic mode ----
    else:
        if config.get("model", "mini") != "deep":
            ensure_mini_model(engine_dir, siril, should_check)

        ckpt_path = (_deep_path() if config.get("model") == "deep"
                     else _mini_path())
        if not ckpt_path.exists():
            print(f"ERROR: model not found at {ckpt_path}", file=sys.stderr)
            sys.exit(1)

        if is_single:
            image = siril.get_image()
            raw   = image.data
            xrgb, orig_dtype, scale, orig_was_mono = _prepare_for_inference(raw)
            padded, orig_hw = _pad_reflect(xrgb, int(config.get("pad", 96)))

            def _prog(v):
                print(f"  [{v:3d}%]", end="\r")

            denoised_padded = process_image(
                padded,
                tile              = int(config.get("tile_size", 512)),
                overlap           = int(config.get("overlap", 64)),
                use_amp           = bool(config.get("use_amp", False)),
                use_gpu           = use_gpu,
                modulation        = int(config.get("modulation", 100)) / 100.0,
                model_path        = ckpt_path,
                progress_callback = _prog,
            )
            print()

            # Ensure HxWx3 for unpadding
            if denoised_padded.ndim == 2:
                denoised_padded = np.stack([denoised_padded] * 3, axis=-1)
            denoised_hwc3 = _unpad(denoised_padded, orig_hw, int(config.get("pad", 96)))

            # Collapse to mono or produce Siril planar layout (RGB, no channel swap)
            if orig_was_mono:
                result = _restore_dtype(
                    denoised_hwc3.mean(axis=2)[np.newaxis, ...],  # (1,H,W)
                    orig_dtype, scale
                )
            else:
                result = _restore_dtype(
                    denoised_hwc3.transpose(2, 0, 1),             # (3,H,W) RGB
                    orig_dtype, scale
                )

            try:
                with siril.image_lock():
                    siril.set_image_pixeldata(result)
                print("Denoised image sent to Siril.")
            except Exception as exc:
                print(f"Could not update Siril image: {exc}")

        else:
            _run_sequence(siril, config, engine_dir, use_gpu)


if __name__ == "__main__":
    main()
