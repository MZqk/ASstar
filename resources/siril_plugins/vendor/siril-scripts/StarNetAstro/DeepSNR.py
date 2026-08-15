# Copyright (C) 2012-2026 team free-astro (see more in AUTHORS file)
# Copyright (C) 2026 StarNetAstro contributors
# Reference site is https://siril.org
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Maintained by StarNetAstro contributors for DeepSNR Siril integration support.
"""
DeepSNR denoising for Siril.

A self-contained wrapper around the third-party DeepSNR (``deepsnr``)
command-line tool. It denoises the currently loaded image or a loaded FITS
sequence and loads the result back into Siril.

The script offers a PyQt6 graphical interface when launched from the Scripts
menu, and a command-line interface for use with the ``pyscript`` command.

DeepSNR is a separate program written by Mikita (Nikita) Misiura
(https://starnetastro.com) and must be installed separately. Its location is
configured inside this script.

Usage (GUI):
    Launch from the Siril Scripts menu.

Usage (CLI, via the pyscript command):
    pyscript DeepSNR.py --linear --model 2
    pyscript DeepSNR.py --sequence --no-linear --stride 256

For FAQ and support guidance, see https://starnetastro.com/faq/.
DeepSNR discussion and help requests can be posted in:
https://www.cloudynights.com/forums/topic/857031-deepsnr-new-dl-based-software-for-noise-reduction/

Original author: Adrian Knagg-Baugh
Current maintainer: StarNetAstro contributors
"""

# Version history
# 1.0.0 - Initial release: DeepSNR support, PyQt6 GUI + CLI, single image and
#         FITS sequence processing, MTF autostretch for linear data.
# 1.1.0 - Maintained by StarNetAstro: move to the StarNetAstro script category
#         and track script/product compatibility separately.

import sirilpy as s

# imagecodecs is required by tifffile to decode DeepSNR's LZW-compressed TIFF
# output (and to write compressed TIFFs).
s.ensure_installed("PyQt6", "numpy", "tifffile", "imagecodecs")

import os
import re
import sys
import json
import math
import time
import shutil
import hashlib
import tempfile
import argparse
import threading
import subprocess
import base64
import platform
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

from sirilpy import LogColor, SequenceType

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QDoubleSpinBox, QRadioButton, QButtonGroup, QProgressBar, QFileDialog,
    QMessageBox, QPlainTextEdit, QDialog, QSlider,
)

SCRIPT_VERSION = "1.1.0"
SUPPORTED_DEEPSNR_MIN_VERSION = "1.2.0"
SUPPORTED_DEEPSNR_MIN_VERSION_TUPLE = (1, 2, 0)
CONFIG_SUBDIR = "DeepSNR"
CONFIG_FILENAME = "settings.json"
DEEPSNR_WEBSITE = "https://starnetastro.com/cli-tools/deepsnr/"
SUPPORT_URL = "https://www.cloudynights.com/forums/topic/857031-deepsnr-new-dl-based-software-for-noise-reduction/"
LATEST_CLI_FEED_URL = "https://starnetastro.com/cli-tools/latest.json"
LATEST_CLI_FEED_SCHEMA = "starnetastro.cli-tools.latest.v1"
LATEST_CLI_CACHE_SECONDS = 24 * 60 * 60
LATEST_CLI_TIMEOUT_SECONDS = 1.0
ICON_B64 = (
    "LyogWFBNICovCnN0YXRpYyBjb25zdCBjaGFyICpEZWVwU05SSWNvbl9YUE1bXSA9IHsKLyogY29s"
    "dW1ucyByb3dzIGNvbG9ycyBjaGFycy1wZXItcGl4ZWwgKi8KIjI0IDI0IDExMyAyICIsCiIgICBj"
    "IE5vbmUiLAoiLiAgYyAjMTExNzFFIiwKIlggIGMgIzEyMTgxRiIsCiJvICBjICMxMjE3MjAiLAoi"
    "TyAgYyAjMTIxQjI0IiwKIisgIGMgIzEyMUUyQSIsCiJAICBjICMxNTFGMzIiLAoiIyAgYyAjMTMy"
    "MDJFIiwKIiQgIGMgIzE0MjIzNCIsCiIlICBjICMxQjIxMzYiLAoiJiAgYyAjMUQyQzM1IiwKIiog"
    "IGMgIzE1MjUzQiIsCiI9ICBjICMxQzIyMzgiLAoiLSAgYyAjMTUyODNDIiwKIjsgIGMgIzFDMkIz"
    "OSIsCiI6ICBjICMyMDIyMzUiLAoiPiAgYyAjMTYyNDQwIiwKIiwgIGMgIzE2MkE0NCIsCiI8ICBj"
    "ICMxOTJBNDQiLAoiMSAgYyAjMTYyRDRCIiwKIjIgIGMgIzE5MkQ0QyIsCiIzICBjICMxNzM2NDYi"
    "LAoiNCAgYyAjMTQzOTQ2IiwKIjUgIGMgIzE3MzA0RSIsCiI2ICBjICMxMzNBNEIiLAoiNyAgYyAj"
    "MTUyRTUwIiwKIjggIGMgIzE4MkY1MSIsCiI5ICBjICMxNzMxNTIiLAoiMCAgYyAjMTkzMzU1IiwK"
    "InEgIGMgIzE5Mzg1NCIsCiJ3ICBjICMxNzMzNUEiLAoiZSAgYyAjMTkzNTVCIiwKInIgIGMgIzFF"
    "M0I1RiIsCiJ0ICBjICMwRDNFNkUiLAoieSAgYyAjMTczNzYzIiwKInUgIGMgIzE5MzY2MSIsCiJp"
    "ICBjICMxMjNDNjciLAoicCAgYyAjMUEzQTYxIiwKImEgIGMgIzFBM0Q2RSIsCiJzICBjICMwRTNF"
    "NzEiLAoiZCAgYyAjMTAzRTc0IiwKImYgIGMgIzFBM0U3MSIsCiJnICBjICMxQjQxNzUiLAoiaCAg"
    "YyAjMTQ0NDdFIiwKImogIGMgIzFBNDE3OSIsCiJrICBjICMzNzRFN0QiLAoibCAgYyAjM0U0QTdG"
    "IiwKInogIGMgIzQxNDk3QyIsCiJ4ICBjICMxOTM2ODMiLAoiYyAgYyAjMTYzQjg2IiwKInYgIGMg"
    "IzE5M0E4MSIsCiJiICBjICMxQTNBOEEiLAoibiAgYyAjMjAzQThDIiwKIm0gIGMgIzE0NDM4MiIs"
    "CiJNICBjICMxNDQ5ODYiLAoiTiAgYyAjMUQ0QTg3IiwKIkIgIGMgIzE3NDc4RCIsCiJWICBjICMx"
    "NTRCOEQiLAoiQyAgYyAjMUQ0RTkzIiwKIlogIGMgIzFENEM5QiIsCiJBICBjICMxRjUyOTUiLAoi"
    "UyAgYyAjMUQ1MTlGIiwKIkQgIGMgIzM0NTA4MCIsCiJGICBjICMzNTREOTciLAoiRyAgYyAjMjA1"
    "Mjk4IiwKIkggIGMgIzM3NTM5NyIsCiJKICBjICMzNTUyOUEiLAoiSyAgYyAjMTQ1M0FCIiwKIkwg"
    "IGMgIzE1NTFCMCIsCiJQICBjICMxNTVCQjYiLAoiSSAgYyAjMTk2NEJEIiwKIlUgIGMgIzIwNjFC"
    "MCIsCiJZICBjICMxNDVGQ0EiLAoiVCAgYyAjMUU2N0MxIiwKIlIgIGMgIzFFNkNDNCIsCiJFICBj"
    "ICMxQzZCRDYiLAoiVyAgYyAjMTE3M0REIiwKIlEgIGMgIzFBNzdFMiIsCiIhICBjICMxMjc4RUUi"
    "LAoifiAgYyAjNEM4NkI3IiwKIl4gIGMgIzQwODdCOSIsCiIvICBjICM0Mjg5QjkiLAoiKCAgYyAj"
    "NTA4NkIyIiwKIikgIGMgIzFEOENGNSIsCiJfICBjICMxRThCRkYiLAoiYCAgYyAjM0U4RkU0IiwK"
    "IicgIGMgIzI5OEVGNSIsCiJdICBjICMyODkwRjMiLAoiWyAgYyAjMjc5MEZGIiwKInsgIGMgIzMy"
    "QTdGRiIsCiJ9ICBjICM0ODg1RDAiLAoifCAgYyAjNEM4OEQ2IiwKIiAuIGMgIzQ1OTNERSIsCiIu"
    "LiBjICM1NEIzRkYiLAoiWC4gYyAjNTdCQkZGIiwKIm8uIGMgIzU2QkNGRSIsCiJPLiBjICM1OUI4"
    "RkUiLAoiKy4gYyAjNUNDMkZGIiwKIkAuIGMgIzVGQzRGRiIsCiIjLiBjICM2NkMzRkYiLAoiJC4g"
    "YyAjNjhDNEZGIiwKIiUuIGMgIzZDQzVGRiIsCiImLiBjICM3MEM5RkYiLAoiKi4gYyAjNzdEMUZG"
    "IiwKIj0uIGMgI0IwRTZGRiIsCiItLiBjICNCNUVERkYiLAoiOy4gYyAjQkZFQkZGIiwKIjouIGMg"
    "I0MzRUVGRiIsCiI+LiBjICNDN0YyRkYiLAoiLC4gYyAjQ0JGNEZGIiwKIjwuIGMgI0Q1RjBGRiIs"
    "CiIxLiBjICNFMEY3RkYiLAoiMi4gYyB3aGl0ZSIsCi8qIHBpeGVscyAqLwoiICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIiwKIiAgLiAuIFggTyBYIE8gTyAr"
    "ICsgIyAjICQgIyArICsgTyBPIE8gTyBPIFggLiAgICIsCiIgIFggWCBPIE8gKyArICsgIyAjICMg"
    "OiAjICMgIyAjICsgKyBPIE8gbyBvIC4gICAiLAoiICBYIE8gTyBPICsgKyAjICMgJCA9IDogOiAk"
    "ICUgJCBAICsgKyBPIE8gTyBvICAgIiwKIiAgTyBPIE8gKyArICQgJCAqICogKiA0IDQgKiAtICQg"
    "JCAkIEAgKyArIE8gTyAgICIsCiIgIE8gTyArICsgJSAkICogLCAsICwgZyBnIDMgLCAsICogJCAk"
    "ICsgKyBPIE8gICAiLAoiICBPICsgQCAlICQgLSAsIDEgOSBxIHggbiByIGUgNSAsIC0gJCAkICsg"
    "TyArICAgIiwKIiAgTyArICMgJCAtIDwgMiB3IGEgbSB6IGwgTSBnIGUgOCAsIC0gJCArICsgTyAg"
    "ICIsCiIgICsgKyAkICogLCAxIGUgZyBOIEsgfiB+IEkgRyBnIGUgNSAsICogJCArICsgICAiLAoi"
    "ICArICsgJSAtIDwgdyBnIEMgVSBXIE8uJS5dIFIgQSBnIDAgLCAtICQgIyArICAgIiwKIiAgKyBA"
    "ICogKiA8IHEgaCBLIFcgeyA9LjwuKy4pIFAgbSAwIDwgPiAkIEAgKyAgICIsCiIgIEAgQCA7IDMg"
    "dCB4IEQgXiBvLi0uMi4yLjouIy59IEYgYyBpIDMgOyAkICMgICAiLAoiICBAICUgOiA1IHQgYyBr"
    "IC8gWC46LjIuMi4sLiYufCBGIGMgaSAzIDsgJSBAICAgIiwKIiAgKyAkICQgPiA8IDAgbSBMICEg"
    "Li48Lj4uKy5bIFkgTSBlIDwgKiAkICMgKyAgICIsCiIgICsgIyAkICogMSBlIGYgQyBUICcgKi4j"
    "Ll8gRSBTIGcgdyAsIC0gJCAjICsgICAiLAoiICArICsgJCAqICwgOCBwIGogRyBJICAuYCBUIFog"
    "aiB5IDUgLCAkICQgIyArICAgIiwKIiAgKyArIEAgPSAqIDIgOCBwIGcgViBIIEogViBmIHkgOSAs"
    "IC0gJCAjICsgTyAgICIsCiIgIE8gKyArICQgKiA+IDEgOSB3IHAgYiBiIHAgZSA3IDEgLSAkICQg"
    "IyArIE8gICAiLAoiICBPIE8gKyArICQgKiAqICwgNSAsIGggcyAyIDEgLSAqICQgJCAjICsgTyBP"
    "ICAgIiwKIiAgTyBPICsgKyArICQgKiAtIC0gKiA2IDYgPiAtICogJCAkICMgKyArIE8gTyAgICIs"
    "CiIgIE8gTyBPICsgKyAjICQgJCAkICQgOyA7ICQgJCAkIEAgKyArICsgKyBYIFggICAiLAoiICBv"
    "IG8gTyBPICsgKyArICMgIyAkID0gKiAkIEAgKyArICsgTyBPIFggWCBPICAgIiwKIiAgLiBYIE8g"
    "TyBPICsgKyAjICsgKyBAIEAgKyArICsgKyBYIE8gTyBYIE8gLiAgICIsCiIgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAiCn07Cg=="
)

# Auto-stretch constants, matched to Siril's defaults (see src/filters/mtf.c).
AS_DEFAULT_SHADOWS_CLIPPING = -2.80
AS_DEFAULT_TARGET_BACKGROUND = 0.25
MAD_NORM = 1.4826

# Output filename prefix (sequence mode).
PREFIX_DENOISED = "denoised_"
DEEPSNR_MODEL_OPTIONS = (
    (2, "v2 (RGB and Grayscale)"),
    (1, "v1 (RGB only)"),
)
DEEPSNR_STRENGTH_TOOLTIP = (
    "Use this setting to reduce the effect:\n"
    "final = strength * denoised + (1 - strength) * original.\n"
    "After a run, adjusting this re-blends instantly without re-running DeepSNR."
)


def normalize_modulation(value):
    """Clamp a user-facing DeepSNR strength value to the modulation range."""
    modulation = float(value)
    if not math.isfinite(modulation):
        return 1.0
    return max(0.0, min(1.0, modulation))


def about_html():
    """Rich-text content for the About dialog."""
    return (
        "<h3>DeepSNR for Siril</h3>"
        f"<p>Script version {SCRIPT_VERSION}</p>"
        "<p>DeepSNR is a third-party neural-network application that reduces "
        "noise in astronomical images, developed by Mikita (Nikita) Misiura. "
        "This script lets Siril hand an image (or sequence) to DeepSNR and load "
        "the denoised result back.</p>"
        f"<p><b>This script supports DeepSNR "
        f"{SUPPORTED_DEEPSNR_MIN_VERSION} or newer.</b></p>"
        "<p>Two models are available. <b>Model 2</b> (the default) works on both "
        "colour and monochrome images. <b>Model 1</b> works on colour (RGB) "
        "images only.</p>"
        "<p>DeepSNR works best on uncorrelated, high-frequency noise. Correlated "
        "noise, such as walking noise, will yield poor results.</p>"
        "<p>Siril loads the source image and provides pixel data to this "
        "script. The current DeepSNR command-line handoff uses temporary "
        "16-bit TIFF files, then the result is loaded back into Siril. The "
        "final Siril image follows the original image sample type unless "
        "Siril is configured to force 16-bit output.</p>"
        "<p>DeepSNR must be installed separately. Please make sure your DeepSNR "
        "installation runs correctly on its own before expecting it to work from "
        "this script. Install the DeepSNR CLI normally so "
        "<code>deepsnr</code> is available on your system PATH or in the "
        "standard installer location. Set the executable location in this "
        "dialog only for nonstandard installations.</p>"
        "<p>If required, the path to the model weights file may also be specified, "
        "however this is usually handled automatically by the DeepSNR packaging.</p>"
        "<p>Help and bug reports: "
        f'<a href="{SUPPORT_URL}">'
        "DeepSNR discussion on Cloudy Nights</a>.</p>"
        f'<p>Website: <a href="{DEEPSNR_WEBSITE}">{DEEPSNR_WEBSITE}</a></p>'
    )


# ---------------------------------------------------------------------------
# Image layout conversion
# ---------------------------------------------------------------------------
#
# Siril hands pixel data to sirilpy as a planar numpy array: shape (H, W) for a
# mono image, or (C, H, W) for a colour image, with dtype uint16 or float32.
# Internally we work in (H, W, C) float32 in the [0, 1] range, which is what the
# numpy maths and tifffile expect.

def siril_to_working(arr):
    """Convert a Siril pixel array to (H, W, C) float32 in [0, 1].

    Returns (working, meta) where meta records how to convert back.
    Float images whose peak exceeds 1.0 are rescaled to avoid clipping the
    highlights when written to an integer TIFF.
    """
    is_uint16 = (arr.dtype == np.uint16)
    if arr.ndim == 2:
        mono = True
        hwc = arr[:, :, np.newaxis].astype(np.float32)
    elif arr.ndim == 3:
        mono = (arr.shape[0] == 1)
        hwc = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported pixel data shape: {arr.shape}")

    rescaled = False
    if is_uint16:
        hwc /= 65535.0
    else:
        scale = 1.0
        peak = float(hwc.max()) if hwc.size else 1.0
        if peak > 1.0:
            scale = peak
            hwc /= scale
            rescaled = True
    hwc = np.clip(hwc, 0.0, 1.0)
    meta = {
        "mono": mono,
        "is_uint16": is_uint16,
        "rescaled": rescaled,
        "scale": 1.0 if is_uint16 else scale,
    }
    return hwc, meta


def working_to_siril(hwc, meta, force_uint16=False):
    """Convert an (H, W, C) float32 [0, 1] array back to Siril's layout/dtype.

    The output dtype matches the original image (meta['is_uint16']) unless
    force_uint16 is set (Siril's core.force_16bit preference), in which case it
    is always saved as 16-bit."""
    hwc = np.clip(hwc, 0.0, 1.0)
    if meta["mono"]:
        out = hwc[:, :, 0]
    else:
        out = np.transpose(hwc, (2, 0, 1))
    if meta["is_uint16"] or force_uint16:
        out = (out * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = out * float(meta.get("scale", 1.0))
        out = out.astype(np.float32)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# TIFF I/O for the DeepSNR round-trip
# ---------------------------------------------------------------------------

def write_deepsnr_tiff(path, hwc):
    """Write an (H, W, C) float [0, 1] array as a 16-bit integer TIFF for DeepSNR."""
    a = hwc[:, :, 0] if hwc.shape[2] == 1 else hwc
    a = (np.clip(a, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(path, a, compression="lzw")


def read_image_tiff(path):
    """Read a TIFF written by DeepSNR, returning (H, W, C) float [0, 1]."""
    a = tifffile.imread(path)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, np.newaxis]
    return np.clip(a, 0.0, 1.0)


# ---------------------------------------------------------------------------
# MTF auto-stretch (faithful to Siril's unlinked autostretch)
# ---------------------------------------------------------------------------
#
# DeepSNR only accepts integer (TIFF/PNG) input. Writing a linear image straight
# to a 16-bit TIFF would crush the signal into the bottom of the range and lose
# precision, so for linear data we autostretch before denoising and reverse the
# stretch afterwards, exactly as the StarNet workflow does.

def _mtf(x, m, lo, hi):
    """Midtones transfer function (scalar or ndarray x)."""
    rng = hi - lo
    xp = np.clip((x - lo) / rng, 0.0, 1.0) if rng > 0 else np.zeros_like(x)
    denom = (2.0 * m - 1.0) * xp - m
    return ((m - 1.0) * xp) / denom


def _mtf_scalar(x, m, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    xp = (x - lo) / (hi - lo)
    return ((m - 1.0) * xp) / ((2.0 * m - 1.0) * xp - m)


def compute_unlinked_mtf_params(hwc):
    """Per-channel (shadows, midtones, highlights), matching Siril's default
    unlinked autostretch (find_unlinked_midtones_balance_default)."""
    nb = hwc.shape[2]
    medians = [float(np.median(hwc[:, :, c])) for c in range(nb)]
    inverted = sum(1 for med in medians if med > 0.5)
    use_inverted = (inverted >= nb)

    params = []
    for c in range(nb):
        chan = hwc[:, :, c]
        median = medians[c]
        mad = float(np.median(np.abs(chan - median))) * MAD_NORM
        if mad == 0.0:
            mad = 0.001
        if not use_inverted:
            c0 = median + AS_DEFAULT_SHADOWS_CLIPPING * mad
            if c0 < 0.0:
                c0 = 0.0
            m2 = median - c0
            midtones = _mtf_scalar(m2, AS_DEFAULT_TARGET_BACKGROUND, 0.0, 1.0)
            params.append((c0, midtones, 1.0))
        else:
            c1 = median - AS_DEFAULT_SHADOWS_CLIPPING * mad
            if c1 > 1.0:
                c1 = 1.0
            m2 = c1 - median
            midtones = 1.0 - _mtf_scalar(m2, AS_DEFAULT_TARGET_BACKGROUND, 0.0, 1.0)
            params.append((0.0, midtones, c1))
    return params


def apply_unlinked_mtf(hwc, params):
    out = np.empty_like(hwc)
    for c, (lo, m, hi) in enumerate(params):
        out[:, :, c] = _mtf(hwc[:, :, c], m, lo, hi)
    return out


def apply_unlinked_pseudoinverse_mtf(hwc, params):
    """Approximate inverse of apply_unlinked_mtf, used to reverse the
    pre-stretch on the denoised image (matches Siril's MTF_pseudoinverse)."""
    out = np.empty_like(hwc)
    for c, (shadows, midtones, highlights) in enumerate(params):
        y = hwc[:, :, c]
        a = (shadows + highlights) * midtones - shadows
        b = shadows * (1.0 - midtones)
        num = a * y + b
        den = (2.0 * midtones - 1.0) * y - midtones + 1.0
        out[:, :, c] = num / den
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# DeepSNR executable: version detection and invocation
# ---------------------------------------------------------------------------

_deepsnr_lock = threading.Lock()
_latest_cli_feed_cache = {"checked_at": 0.0, "payload": None}

# Names to look for on PATH when no explicit executable is configured.
DEEPSNR_EXE_NAMES = ("deepsnr", "deepsnr.exe", "DeepSNR", "DeepSNR.exe")
DEEPSNR_DEFAULT_EXE_PATHS = (
    r"C:\Program Files\DeepSNR\bin\deepsnr.exe",
    "/usr/local/bin/deepsnr",
    "/usr/bin/deepsnr",
)


class DeepSNRError(Exception):
    pass


def find_deepsnr_on_path():
    """Return the path to a DeepSNR executable on PATH, or None."""
    for name in DEEPSNR_EXE_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_deepsnr_installed_default():
    """Return a DeepSNR executable from a known installer path, or None."""
    for path in DEEPSNR_DEFAULT_EXE_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def resolve_exe(configured):
    """Resolve the DeepSNR executable to use.

    PATH and known CLI installer locations are preferred. A configured path is a
    last-resort fallback for nonstandard installations. Returns a path string,
    or None if nothing is found.
    """
    def configured_exe():
        if not configured:
            return None
        candidate = os.path.expanduser(configured.strip())
        if not candidate:
            return None
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return shutil.which(candidate)

    return find_deepsnr_on_path() or find_deepsnr_installed_default() or configured_exe()


def display_exe_path(exe):
    """Format an executable path for UI display without changing execution."""
    if not exe:
        return ""
    path = os.path.normpath(exe)
    root, ext = os.path.splitext(path)
    if ext.lower() == ".exe":
        return root
    return path


def parse_version_tuple(text):
    if not text:
        return None
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text))
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def format_version(version):
    return ".".join(str(part) for part in version)


def current_latest_cli_platform_key(system=None, machine=None):
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    if system.startswith("win"):
        return "windows-x64"
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        return "macos-x64"
    if system == "linux":
        return "linux-x64"
    return None


def latest_cli_platform_label(platform_key):
    return {
        "windows-x64": "Windows",
        "linux-x64": "Linux",
        "macos-x64": "macOS Intel",
        "macos-arm64": "macOS Apple silicon",
    }.get(platform_key, platform_key or "this platform")


def fetch_latest_cli_feed(url=LATEST_CLI_FEED_URL):
    now = time.time()
    cached = _latest_cli_feed_cache.get("payload")
    checked_at = _latest_cli_feed_cache.get("checked_at", 0.0)
    if cached is not None and now - checked_at < LATEST_CLI_CACHE_SECONDS:
        return cached
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"StarNetAstro-Siril/{SCRIPT_VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=LATEST_CLI_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    _latest_cli_feed_cache["checked_at"] = now
    _latest_cli_feed_cache["payload"] = payload
    return payload


def latest_cli_update_notice(product_key, installed_version, *,
                             platform_key=None, fetcher=None):
    if installed_version is None:
        return None
    platform_key = platform_key or current_latest_cli_platform_key()
    if not platform_key:
        return None
    try:
        payload = (fetcher or fetch_latest_cli_feed)()
    except (OSError, ValueError, urllib.error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != LATEST_CLI_FEED_SCHEMA:
        return None
    tool = payload.get("tools", {}).get(product_key)
    if not isinstance(tool, dict):
        return None
    latest = tool.get("latest", {}).get(platform_key)
    if not isinstance(latest, dict):
        return None
    latest_version = parse_version_tuple(latest.get("version"))
    if latest_version is None or latest_version <= installed_version:
        return None
    display_name = tool.get("display_name") or "DeepSNR"
    download_url = tool.get("download_url") or DEEPSNR_WEBSITE
    return (
        f"Update available: {display_name} {format_version(latest_version)} "
        f"is available for {latest_cli_platform_label(platform_key)}.\n"
        f"Download: {download_url}"
    )


def missing_deepsnr_message(location_hint, configured=None):
    prefix = (
        f"DeepSNR executable not found or not executable: {configured.strip()}"
        if configured and configured.strip()
        else "DeepSNR ('deepsnr') was not found in a standard CLI installer "
             "location or on PATH."
    )
    return (
        f"{prefix}\n"
        f"Download the DeepSNR CLI installer from {DEEPSNR_WEBSITE} "
        f"and run the installer.\n"
        f"Alternatively, use {location_hint} to select an existing executable."
    )


def detect_deepsnr_version_info(configured):
    """Probe the configured (or PATH-resolved) DeepSNR executable.

    Returns (status, message, resolved_exe) where status is one of:
        'ok'       - a usable DeepSNR executable was found
        'upgrade'  - an older DeepSNR executable was found
        'missing'  - no executable could be found
        'unknown'  - ran but DeepSNR could not be identified
    """
    exe = resolve_exe(configured)
    if not exe:
        if configured and configured.strip():
            return ("missing",
                    missing_deepsnr_message("Set paths", configured),
                    None)
        return ("missing",
                missing_deepsnr_message("Set paths"),
                None)

    exe_dir = os.path.dirname(os.path.abspath(exe))
    blobs = []
    with _deepsnr_lock:
        # --help carries the clean banner ("DeepSNR v1.2.1", "ONNX Runtime
        # backend"); --version uses TCLAP's "<prog>  version: 1.2.1" form.
        for extra in (["--help"], ["--version"], []):
            try:
                proc = subprocess.run(
                    [exe] + extra, cwd=exe_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=30, text=True, errors="replace",
                )
                blobs.append(proc.stdout or "")
            except subprocess.TimeoutExpired:
                blobs.append("")
            except OSError as exc:
                return "missing", f"Could not run the DeepSNR executable: {exc}", exe
    blob = "\n".join(blobs)
    low = blob.lower()

    if "deepsnr" in low:
        match = re.search(r"deepsnr[\s_]+v?(\d+)\.(\d+)(?:\.(\d+))?", low)
        if not match:
            match = re.search(r"version:\s*v?(\d+)\.(\d+)(?:\.(\d+))?", low)
        if match:
            version = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3) or 0),
            )
            version_text = ".".join(str(part) for part in version)
            if version >= SUPPORTED_DEEPSNR_MIN_VERSION_TUPLE:
                return (
                    "ok",
                    f"DeepSNR {version_text} found\n"
                    f"Path: {display_exe_path(exe)}",
                    exe,
                )
            return ("upgrade",
                    f"DeepSNR {version_text} was detected. This script requires "
                    f"DeepSNR {SUPPORTED_DEEPSNR_MIN_VERSION} or newer - please upgrade.",
                    exe)
        return "ok", f"DeepSNR found\nPath: {display_exe_path(exe)}", exe

    return ("unknown",
            f"The executable at {exe} did not report a recognisable DeepSNR "
            "version.",
            exe)


def detect_deepsnr_version(configured):
    status, message, _exe = detect_deepsnr_version_info(configured)
    return status, message


def build_deepsnr_argv(exe, in_tiff, out_tiff, *, stride=None, model=2,
                       weights=None):
    argv = [exe, "-i", in_tiff, "-o", out_tiff]
    if stride is not None:
        argv += ["-s", str(stride)]
    if model is not None:
        argv += ["-m", str(model)]
    if weights:
        argv += ["-w", weights]
    return argv


def run_deepsnr(exe, in_tiff, out_tiff, *, stride=None, model=2, weights=None,
                progress_cb=None, log_cb=None, cancel_cb=None):
    """Run DeepSNR on in_tiff, producing out_tiff. Raises DeepSNRError on
    failure. Returns True, or 'cancelled'."""
    exe_dir = os.path.dirname(os.path.abspath(exe))
    argv = build_deepsnr_argv(
        exe, in_tiff, out_tiff,
        stride=stride, model=model, weights=weights,
    )

    pct_re = re.compile(rb"(\d{1,3}(?:\.\d+)?)\s*%")
    output_tail = []

    def remember_output(token):
        text = token.decode("utf-8", "replace").strip()
        if not text:
            return
        output_tail.append(text)
        del output_tail[:-10]

    with _deepsnr_lock:
        try:
            proc = subprocess.Popen(
                argv, cwd=exe_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise DeepSNRError(f"Could not launch DeepSNR: {exc}") from exc

        def reader():
            fd = proc.stdout.fileno()
            pending = b""
            while True:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                pending += data
                parts = re.split(rb"[\r\n]+", pending)
                pending = parts.pop()
                for token in parts:
                    last = None
                    for m in pct_re.finditer(token):
                        last = m
                    if last and progress_cb:
                        try:
                            progress_cb(min(1.0, float(last.group(1)) / 100.0))
                        except Exception:
                            pass
                    # Progress lines (containing a percentage) drive the bar only;
                    # don't echo them to the log. Log other informational lines.
                    if last is None:
                        remember_output(token)
                        if log_cb:
                            log_cb(output_tail[-1])
            if pending:
                last = None
                for m in pct_re.finditer(pending):
                    last = m
                if last and progress_cb:
                    try:
                        progress_cb(min(1.0, float(last.group(1)) / 100.0))
                    except Exception:
                        pass
                if last is None:
                    remember_output(pending)
                    if log_cb:
                        log_cb(output_tail[-1])

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        cancelled = False
        while proc.poll() is None:
            if cancel_cb and cancel_cb():
                cancelled = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(0.1)
        thread.join(timeout=2)

    if cancelled:
        return "cancelled"
    if proc.returncode != 0:
        detail = ""
        if output_tail:
            detail = "\nLast DeepSNR output:\n" + "\n".join(output_tail)
        raise DeepSNRError(f"DeepSNR exited with status {proc.returncode}.{detail}")
    if not os.path.isfile(out_tiff):
        raise DeepSNRError("DeepSNR did not produce an output file.")
    return True


# ---------------------------------------------------------------------------
# Core processing: one image
# ---------------------------------------------------------------------------

class DeepSNROptions:
    """Plain options container shared by the GUI and the CLI."""

    def __init__(self):
        self.exe = ""
        self.weights = ""
        self.linear = True
        self.custom_stride = False
        self.stride = 256
        self.model = 2  # 1 (RGB only) or 2 (RGB and mono)
        # Blend factor between original (0.0) and fully denoised (1.0). Not
        # persisted: it always starts at 1.0 each time the script runs.
        self.modulation = 1.0

    def stride_value(self):
        if not self.custom_stride:
            return None
        v = int(self.stride)
        if v % 2:
            v += 1
        return max(2, min(512, v))

    def to_dict(self):
        return {
            "exe": self.exe, "weights": self.weights, "linear": self.linear,
            "custom_stride": self.custom_stride, "stride": self.stride,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d):
        o = cls()
        o.exe = d.get("exe", "")
        o.weights = d.get("weights", "")
        o.linear = bool(d.get("linear", True))
        o.custom_stride = bool(d.get("custom_stride", False))
        o.stride = int(d.get("stride", 256))
        o.model = 1 if int(d.get("model", 2)) == 1 else 2
        return o


def ai_signature(opts):
    """Tuple of the parameters that affect the DeepSNR (AI) output. Used as a
    cache key: if these are unchanged, only the modulation blend needs redoing."""
    return (opts.model, opts.stride_value(), (opts.weights or "").strip(),
            bool(opts.linear))


def fingerprint(arr):
    """Cheap content fingerprint of a pixel array, to detect a changed image."""
    return (tuple(arr.shape), str(arr.dtype),
            hashlib.blake2b(np.ascontiguousarray(arr).tobytes(),
                            digest_size=8).hexdigest())


def arr_naxes(arr):
    """Siril (w, h, c) dimensions for a Siril-layout pixel array."""
    if arr.ndim == 2:
        return (arr.shape[1], arr.shape[0], 1)
    return (arr.shape[2], arr.shape[1], arr.shape[0])


def validate_working_shape(hwc, *, model):
    channels = hwc.shape[2]
    if channels not in (1, 3):
        raise DeepSNRError(
            f"DeepSNR requires mono or RGB input; got {channels} channel(s).")
    if model == 1 and channels != 3:
        raise DeepSNRError("DeepSNR Model 1 requires a colour (RGB) image.")


def validate_deepsnr_output_shape(denoised, original):
    if denoised.shape[:2] != original.shape[:2]:
        raise DeepSNRError(
            "DeepSNR output dimensions do not match the input image "
            f"({denoised.shape[:2]} vs {original.shape[:2]}).")
    out_channels = denoised.shape[2]
    in_channels = original.shape[2]
    if out_channels not in (1, 3):
        raise DeepSNRError(
            f"DeepSNR output has unsupported {out_channels}-channel data.")
    if in_channels == 3 and out_channels != 3:
        raise DeepSNRError("DeepSNR returned mono data for an RGB input image.")


def cache_decision(cache, last_written_fp, opts, cur_arr):
    """Decide how a single-image run should proceed, given the cache and the
    image currently loaded in Siril.

    Returns one of:
        ('reblend', None)        - only the modulation changed; re-blend the
                                   cached result, no model re-run needed
        ('rerun', original_arr)  - run the model on original_arr (the cached
                                   true original if the displayed image is our
                                   own output, else the freshly loaded image)
    """
    sig = ai_signature(opts)
    # Is the image loaded in Siril still our last output? If so the cached
    # original is the true source, not the displayed (denoised) image.
    ours = (cache is not None and last_written_fp is not None
            and fingerprint(cur_arr) == last_written_fp)
    if ours and cache["ai_sig"] == sig:
        return ("reblend", None)
    return ("rerun", cache["original_arr"] if ours else cur_arr)


def denoise_image(arr, opts, force_16bit=False, progress_cb=None,
                  log_cb=None, cancel_cb=None):
    """Run the DeepSNR (AI) step on a single Siril pixel array.

    Returns (original_hwc, denoised_hwc, meta) where both arrays are float32
    [0, 1] in the original image domain (HWC layout), ready to be blended; or
    None if cancelled. This is the expensive step and is what the GUI caches.
    """
    exe = resolve_exe(opts.exe)
    if not exe:
        raise DeepSNRError(missing_deepsnr_message("--exe"))

    working, meta = siril_to_working(arr)
    validate_working_shape(working, model=opts.model)
    if opts.model == 1 and meta["mono"]:
        raise DeepSNRError(
            "DeepSNR Model 1 requires a colour (RGB) image. Use Model 2 for "
            "monochrome images.")
    if meta["rescaled"] and log_cb:
        log_cb("Pixel values exceed 1.0; rescaling to avoid clipping the highlights.")
    original_hwc = working.copy()

    # Pre-stretch linear data so DeepSNR sees a well-conditioned 16-bit image.
    params = None
    if opts.linear:
        params = compute_unlinked_mtf_params(working)
        working = apply_unlinked_mtf(working, params)
        if log_cb:
            log_cb("Linear mode: applied MTF autostretch before denoising.")

    tmp = tempfile.mkdtemp(prefix="siril_deepsnr_")
    try:
        in_tiff = os.path.join(tmp, "input.tif")
        out_tiff = os.path.join(tmp, "denoised.tif")

        write_deepsnr_tiff(in_tiff, working)

        result = run_deepsnr(
            exe, in_tiff, out_tiff,
            stride=opts.stride_value(), model=opts.model,
            weights=(opts.weights or None),
            progress_cb=progress_cb, log_cb=log_cb, cancel_cb=cancel_cb,
        )
        if result == "cancelled":
            return None

        denoised = read_image_tiff(out_tiff)
        validate_deepsnr_output_shape(denoised, original_hwc)
        if meta["mono"] and denoised.shape[2] == 3:
            denoised = denoised[:, :, :1]

        # Reverse the pre-stretch so the result is back in the original domain.
        if opts.linear and params is not None:
            denoised = apply_unlinked_pseudoinverse_mtf(denoised, params)

        return original_hwc, denoised, meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def blend_result(original_hwc, denoised_hwc, modulation, meta, force_16bit=False):
    """Blend the original and denoised images: modulation 1.0 is fully denoised,
    0.0 is the untouched original. Returns a Siril-layout array."""
    mod = float(max(0.0, min(1.0, modulation)))
    blended = (1.0 - mod) * original_hwc + mod * denoised_hwc
    return working_to_siril(blended, meta, force_16bit)


def process_array(arr, opts, force_16bit=False, progress_cb=None,
                  log_cb=None, cancel_cb=None):
    """Denoise a single Siril pixel array and apply the modulation blend.

    Returns the result as a Siril-layout array (same dtype as the input unless
    force_16bit is set), or None if cancelled.
    """
    result = denoise_image(arr, opts, force_16bit, progress_cb, log_cb, cancel_cb)
    if result is None:
        return None
    original_hwc, denoised_hwc, meta = result
    return blend_result(original_hwc, denoised_hwc, opts.modulation, meta, force_16bit)


# ---------------------------------------------------------------------------
# Siril-facing flows
# ---------------------------------------------------------------------------

def _get_config(siril, group, key, default=None):
    try:
        value = siril.get_siril_config(group, key)
        return default if value is None else value
    except Exception:
        return default


def _config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    return default


def _force_16bit(siril):
    return _config_bool(_get_config(siril, "core", "force_16bit", False))


def selected_sequence_indices(seq):
    imgparam = getattr(seq, "imgparam", None)
    if imgparam:
        selected = [
            i for i, params in enumerate(imgparam[:seq.number])
            if getattr(params, "incl", True)
        ]
        if selected:
            return selected
    return list(range(seq.number))


def process_single_image(siril, opts, progress_cb=None, log_cb=None,
                         cancel_cb=None):
    """Denoise the loaded image and load the result back into Siril."""
    arr = siril.get_image_pixeldata()
    result = process_array(arr, opts, force_16bit=_force_16bit(siril),
                           progress_cb=progress_cb, log_cb=log_cb,
                           cancel_cb=cancel_cb)
    if result is None:
        return False

    siril.undo_save_state(f"DeepSNR (model {opts.model})")
    with siril.image_lock():
        siril.set_image_pixeldata(result)
    return True


def process_sequence(siril, opts, progress_cb=None, log_cb=None,
                     cancel_cb=None):
    """Denoise every selected frame of the loaded FITS sequence, building a new
    'denoised_' sequence."""
    seq = siril.get_seq()
    if seq.type != SequenceType.SEQ_REGULAR:
        raise DeepSNRError(
            "Sequence processing requires a FITS sequence (SER / FITSEQ are not "
            "supported by this script). Convert the sequence to FITS first.")

    indices = selected_sequence_indices(seq)
    n = len(indices)
    force_16bit = _force_16bit(siril)
    if log_cb:
        total = seq.number
        if n == total:
            log_cb(f"Processing {n} frame(s) from sequence '{seq.seqname}'.")
        else:
            log_cb(f"Processing {n}/{total} selected frame(s) from sequence '{seq.seqname}'.")

    for pos, i in enumerate(indices):
        if cancel_cb and cancel_cb():
            return False
        if log_cb:
            log_cb(f"Frame {pos + 1}/{n} (sequence index {i + 1})")

        def frame_progress(frac, base=pos):
            if progress_cb:
                progress_cb((base + frac) / n)

        arr = siril.get_seq_frame_pixeldata(i)
        result = process_array(arr, opts, force_16bit=force_16bit,
                               progress_cb=frame_progress, log_cb=None,
                               cancel_cb=cancel_cb)
        if result is None:
            return False

        siril.set_seq_frame_pixeldata(i, result, prefix=PREFIX_DENOISED)
        if progress_cb:
            progress_cb((pos + 1) / n)

    siril.create_new_seq(PREFIX_DENOISED)
    if log_cb:
        log_cb("Sequence processing complete.")
    return True


# ---------------------------------------------------------------------------
# Settings persistence (self-contained, in the script's own config dir)
# ---------------------------------------------------------------------------

def settings_path(siril):
    cfg_dir = Path(siril.get_siril_configdir()) / CONFIG_SUBDIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / CONFIG_FILENAME


def load_options(siril):
    try:
        data = json.loads(settings_path(siril).read_text(encoding="utf-8"))
        return DeepSNROptions.from_dict(data)
    except FileNotFoundError:
        return DeepSNROptions()
    except Exception:
        return DeepSNROptions()


def save_options(siril, opts):
    try:
        settings_path(siril).write_text(
            json.dumps(opts.to_dict(), indent=2), encoding="utf-8")
    except Exception as exc:
        siril.log(f"DeepSNR: could not save settings: {exc}", LogColor.SALMON)


# ---------------------------------------------------------------------------
# PyQt6 worker thread
# ---------------------------------------------------------------------------

class DeepSNRWorker(QThread):
    progress = pyqtSignal(float)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(bool, str)

    def __init__(self, siril, opts, sequence_mode, arr=None):
        super().__init__()
        self.siril = siril
        self.opts = opts
        self.sequence_mode = sequence_mode
        self.arr = arr  # pre-fetched original (single-image mode)
        self.cache_payload = None  # (original_hwc, denoised_hwc, meta) on success
        self.written_fp = None  # fingerprint of the array pushed to Siril
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self):
        return self._cancel

    def run(self):
        try:
            if self.sequence_mode:
                ok = process_sequence(
                    self.siril, self.opts,
                    progress_cb=self.progress.emit, log_cb=self.message.emit,
                    cancel_cb=self._cancelled)
                if self._cancel:
                    self.finished_ok.emit(False, "Cancelled.")
                else:
                    self.finished_ok.emit(bool(ok),
                                          "DeepSNR finished." if ok else "DeepSNR did not complete.")
                return

            # Single image: run the AI step, keep its result for caching, then
            # apply the modulation blend.
            arr = self.arr if self.arr is not None else self.siril.get_image_pixeldata()
            force_16bit = _force_16bit(self.siril)
            result = denoise_image(
                arr, self.opts, force_16bit,
                progress_cb=self.progress.emit, log_cb=self.message.emit,
                cancel_cb=self._cancelled)
            if result is None or self._cancel:
                self.finished_ok.emit(False, "Cancelled.")
                return
            original_hwc, denoised_hwc, meta = result
            self.cache_payload = (original_hwc, denoised_hwc, meta)
            blended = blend_result(original_hwc, denoised_hwc,
                                   self.opts.modulation, meta, force_16bit)
            self.written_fp = fingerprint(blended)
            self.siril.undo_save_state(f"DeepSNR (model {self.opts.model})")
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(blended)
            self.finished_ok.emit(True, "DeepSNR finished.")
        except Exception as exc:
            self.finished_ok.emit(False, str(exc))


class LatestCliVersionWorker(QThread):
    notice = pyqtSignal(str)

    def __init__(self, installed_version):
        super().__init__()
        self.installed_version = installed_version

    def run(self):
        notice = latest_cli_update_notice("deepsnr", self.installed_version)
        if notice:
            self.notice.emit(notice)


def deepsnr_window_icon():
    pixmap = QPixmap()
    if pixmap.loadFromData(base64.b64decode(ICON_B64), "XPM"):
        return QIcon(pixmap)
    return QIcon()


# ---------------------------------------------------------------------------
# PyQt6 GUI
# ---------------------------------------------------------------------------

class DeepSNRGUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.opts = load_options(siril)
        self.worker = None
        # Cached AI result so a modulation-only change can be re-blended without
        # re-running the model. Keys: ai_sig, original_arr (the true original
        # input), denoised_hwc. _last_written_fp is the fingerprint of the array
        # most recently pushed to Siril, used to tell whether the image loaded in
        # Siril is still our output (cache valid) or has been replaced.
        self._cache = None
        self._last_written_fp = None
        self._pending_sig = None
        self._pending_arr = None
        self.update_worker = None
        self.has_image = False
        self.has_sequence = False
        try:
            self.has_image = siril.is_image_loaded()
        except Exception:
            pass
        try:
            self.has_sequence = siril.is_sequence_loaded()
        except Exception:
            pass

        self.setWindowTitle("DeepSNR denoising")
        self._icon = deepsnr_window_icon()
        self.setWindowIcon(self._icon)
        self._build_ui()
        self._load_into_ui()
        self._check_executable()

    # -- UI construction --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.exe_path = ""
        self.weights_path = ""
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.update_label = QLabel("")
        self.update_label.setWordWrap(True)
        self.update_label.setStyleSheet("color: #d99200;")
        layout.addWidget(self.update_label)
        status_line = QLabel("")
        status_line.setFixedHeight(1)
        status_line.setStyleSheet("border-top: 1px solid gray;")
        layout.addWidget(status_line)

        primary_layout = QGridLayout()
        primary_layout.setColumnStretch(1, 1)

        strength_label = QLabel("Strength:")
        strength_label.setToolTip(DEEPSNR_STRENGTH_TOOLTIP)
        strength_row = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setToolTip(DEEPSNR_STRENGTH_TOOLTIP)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setSingleStep(1)
        self.strength_slider.setPageStep(10)
        self.strength_slider.valueChanged.connect(self._on_strength_slider_changed)
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setToolTip(DEEPSNR_STRENGTH_TOOLTIP)
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setSingleStep(0.01)
        self.strength_spin.setKeyboardTracking(False)
        self.strength_spin.valueChanged.connect(self._on_strength_spin_changed)
        strength_row.addWidget(self.strength_slider, 1)
        strength_row.addWidget(self.strength_spin)
        primary_layout.addWidget(strength_label, 0, 0)
        primary_layout.addLayout(strength_row, 0, 1)

        model_label = QLabel("Model version:")
        model_label.setToolTip("Selects model version.")
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Selects model version.")
        for model, label in DEEPSNR_MODEL_OPTIONS:
            self.model_combo.addItem(label, model)
        primary_layout.addWidget(model_label, 1, 0)
        primary_layout.addWidget(self.model_combo, 1, 1)
        layout.addLayout(primary_layout)

        # Options
        opt_group = QGroupBox("Processing Options")
        opt_layout = QVBoxLayout(opt_group)

        self.linear_check = QCheckBox("Linear data")
        self.linear_check.setToolTip(
            "Enable when the loaded image is unstretched (linear). An automatic "
            "stretch is applied before denoising and reversed afterwards, which "
            "preserves precision. Disable if the image has already been stretched.")
        opt_layout.addWidget(self.linear_check)
        layout.addWidget(opt_group)

        # Target
        target_group = QGroupBox("Apply to")
        target_layout = QHBoxLayout(target_group)
        self.target_image = QRadioButton("Loaded image")
        self.target_seq = QRadioButton("Loaded sequence")
        self.target_image.setToolTip("Process the single image currently loaded in Siril.")
        self.target_seq.setToolTip("Process every selected frame of the loaded FITS sequence.")
        self.target_group = QButtonGroup(self)
        self.target_group.addButton(self.target_image)
        self.target_group.addButton(self.target_seq)
        target_layout.addWidget(self.target_image)
        target_layout.addWidget(self.target_seq)
        self.target_image.setEnabled(self.has_image)
        self.target_seq.setEnabled(self.has_sequence)
        if self.has_sequence and not self.has_image:
            self.target_seq.setChecked(True)
        else:
            self.target_image.setChecked(True)
        layout.addWidget(target_group)

        # Progress + log
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setFixedHeight(120)
        layout.addWidget(self.log_view)

        # Buttons
        btn_row = QHBoxLayout()
        about_btn = QPushButton("About")
        about_btn.setToolTip("About DeepSNR and this script.")
        about_btn.clicked.connect(self._show_about)
        paths_btn = QPushButton("Set paths")
        paths_btn.setToolTip(
            "Optional executable and weights paths for nonstandard installs.")
        paths_btn.clicked.connect(self._show_paths)
        self.run_btn = QPushButton("Run DeepSNR")
        self.run_btn.clicked.connect(self._on_run)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(about_btn)
        btn_row.addWidget(paths_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    # -- settings <-> widgets --------------------------------------------
    def _load_into_ui(self):
        self.exe_path = self.opts.exe
        self.weights_path = self.opts.weights
        self.linear_check.setChecked(self.opts.linear)
        model_index = self.model_combo.findData(self.opts.model)
        if model_index < 0:
            model_index = self.model_combo.findData(2)
        self.model_combo.setCurrentIndex(model_index)
        # Modulation always starts fully denoised (1.0); it is never persisted.
        self._set_modulation_value(1.0, notify=False)

    def _modulation_value(self):
        return normalize_modulation(self.strength_spin.value())

    def _set_modulation_value(self, value, notify=True):
        modulation = normalize_modulation(value)
        slider_value = int(round(modulation * 100))
        old_slider_block = self.strength_slider.blockSignals(True)
        old_spin_block = self.strength_spin.blockSignals(True)
        self.strength_slider.setValue(slider_value)
        self.strength_spin.setValue(slider_value / 100.0)
        self.strength_slider.blockSignals(old_slider_block)
        self.strength_spin.blockSignals(old_spin_block)
        if notify:
            self._on_modulation_changed()

    def _on_strength_slider_changed(self, value):
        self._set_modulation_value(value / 100.0)

    def _on_strength_spin_changed(self, value):
        self._set_modulation_value(value)

    def _collect_options(self):
        self.opts.exe = self.exe_path.strip()
        self.opts.weights = self.weights_path.strip()
        self.opts.linear = self.linear_check.isChecked()
        self.opts.custom_stride = False
        self.opts.stride = 256
        self.opts.model = int(self.model_combo.currentData() or 2)
        self.opts.modulation = self._modulation_value()
        return self.opts

    # -- modulation / cached re-blend ------------------------------------
    def _image_naxes(self):
        try:
            return tuple(self.siril.get_image(with_pixels=False).naxes)
        except Exception:
            return None

    def _apply_modulation(self):
        """Re-blend the cached AI result at the current modulation and push it to
        Siril. Returns True on success, False if there is no usable cache."""
        cache = self._cache
        if not cache:
            return False
        # Guard against a different-sized image having been loaded since the
        # cache was built (cheap dimension check, no pixel transfer).
        dims = self._image_naxes()
        if dims is not None and dims != arr_naxes(cache["original_arr"]):
            self._cache = None
            self._last_written_fp = None
            return False
        original_hwc, meta = siril_to_working(cache["original_arr"])
        blended = blend_result(original_hwc, cache["denoised_hwc"],
                               self._modulation_value(), meta,
                               _force_16bit(self.siril))
        try:
            with self.siril.image_lock():
                self.siril.set_image_pixeldata(blended)
        except Exception as exc:
            self._on_message(f"Could not apply modulation: {exc}")
            return False
        self._last_written_fp = fingerprint(blended)
        return True

    def _on_modulation_changed(self):
        # Only re-blend live when a cached result exists for the current AI
        # settings (single-image mode) and nothing is running.
        if self.target_seq.isChecked() or not self._cache:
            return
        if self.worker and self.worker.isRunning():
            return
        if self._cache["ai_sig"] != ai_signature(self._collect_options()):
            return
        # Only overwrite the image if it is still our last output.
        try:
            cur = self.siril.get_image_pixeldata()
        except Exception:
            return
        if self._last_written_fp is None or fingerprint(cur) != self._last_written_fp:
            return
        if self._apply_modulation():
            self._on_message(
                f"Denoise strength {self._modulation_value():.2f} applied "
                "(no re-run needed).")

    # -- about ------------------------------------------------------------
    def _show_about(self):
        dlg = QDialog(self)
        dlg.setWindowIcon(self._icon)
        dlg.setWindowTitle("About DeepSNR")
        lay = QVBoxLayout(dlg)
        label = QLabel(about_html())
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        lay.addWidget(label)
        row = QHBoxLayout()
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        row.addWidget(close)
        lay.addLayout(row)
        dlg.exec()

    # -- executable handling ---------------------------------------------
    def _browse_exe(self, target=None):
        path, _ = QFileDialog.getOpenFileName(self, "Select the DeepSNR executable")
        if path:
            if target is not None:
                target.setText(path)
            else:
                self.exe_path = path
                self._check_executable()

    def _browse_weights(self, target=None):
        path, _ = QFileDialog.getOpenFileName(self, "Select the DeepSNR weights file")
        if path:
            if target is not None:
                target.setText(path)
            else:
                self.weights_path = path

    def _show_paths(self):
        dlg = QDialog(self)
        dlg.setWindowIcon(self._icon)
        dlg.setWindowTitle("Set paths")
        dlg.resize(900, 220)
        lay = QVBoxLayout(dlg)
        form = QGridLayout()
        form.setColumnStretch(1, 1)

        exe_edit = QLineEdit()
        exe_edit.setMinimumWidth(560)
        exe_edit.setText(self.exe_path or getattr(self, "_resolved_exe_path", ""))
        exe_edit.setPlaceholderText(
            "Normally leave blank; use only for nonstandard installs")
        exe_edit.setToolTip(
            "Optional path to a nonstandard DeepSNR executable. Normally leave "
            "blank so the script uses PATH or the standard CLI installer "
            "location.")
        exe_browse = QPushButton("Browse...")
        exe_browse.clicked.connect(lambda: self._browse_exe(exe_edit))
        form.addWidget(QLabel("Executable:"), 0, 0)
        form.addWidget(exe_edit, 0, 1)
        form.addWidget(exe_browse, 0, 2)

        weights_edit = QLineEdit()
        weights_edit.setMinimumWidth(560)
        weights_edit.setText(self.weights_path)
        weights_edit.setPlaceholderText(
            "Optional - leave blank to use the bundled weights")
        weights_edit.setToolTip(
            "Optional path to an alternative model weights file. Leave blank to "
            "use the weights bundled with DeepSNR.")
        weights_browse = QPushButton("Browse...")
        weights_browse.clicked.connect(lambda: self._browse_weights(weights_edit))
        weights_clear = QPushButton("Clear")
        weights_clear.clicked.connect(lambda: weights_edit.clear())
        form.addWidget(QLabel("Weights:"), 1, 0)
        form.addWidget(weights_edit, 1, 1)
        wbox = QHBoxLayout()
        wbox.addWidget(weights_browse)
        wbox.addWidget(weights_clear)
        wcell = QWidget()
        wcell.setLayout(wbox)
        form.addWidget(wcell, 1, 2)
        lay.addLayout(form)

        row = QHBoxLayout()
        row.addStretch()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        lay.addLayout(row)

        if dlg.exec():
            self.exe_path = exe_edit.text().strip()
            self.weights_path = weights_edit.text().strip()
            self._check_executable()

    def _check_executable(self):
        exe = self.exe_path.strip()
        status, message, resolved_exe = detect_deepsnr_version_info(exe)
        self._resolved_exe_path = resolved_exe or ""
        colors = {"ok": "green", "upgrade": "darkorange",
                  "missing": "red", "unknown": "darkorange"}
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {colors.get(status, 'black')};")
        self.update_label.setText("")
        self._exe_ok = (status == "ok")
        if self._exe_ok:
            self._start_latest_cli_check(parse_version_tuple(message))
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(self._exe_ok)
        return self._exe_ok

    def _start_latest_cli_check(self, installed_version):
        if installed_version is None:
            return
        self.update_worker = LatestCliVersionWorker(installed_version)
        self.update_worker.notice.connect(self._show_latest_cli_notice)
        self.update_worker.start()

    def _show_latest_cli_notice(self, notice):
        self.update_label.setText(notice)

    # -- run --------------------------------------------------------------
    def _on_run(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.run_btn.setEnabled(False)
            return

        opts = self._collect_options()
        if not self._check_executable():
            return
        sequence_mode = self.target_seq.isChecked()
        if sequence_mode and not self.has_sequence:
            QMessageBox.warning(self, "No sequence", "No sequence is loaded.")
            return
        if not sequence_mode and not self.has_image:
            QMessageBox.warning(self, "No image", "No image is loaded.")
            return

        save_options(self.siril, opts)

        original_arr = None
        if not sequence_mode:
            try:
                cur = self.siril.get_image_pixeldata()
            except Exception as exc:
                QMessageBox.warning(self, "DeepSNR", f"Could not read the image: {exc}")
                return
            decision, original_arr = cache_decision(
                self._cache, self._last_written_fp, opts, cur)
            if decision == "reblend" and self._apply_modulation():
                self._on_message(
                    f"Denoise strength {self._modulation_value():.2f} applied "
                    "(cached result, model not re-run).")
                return
            if original_arr is None:  # reblend fell through (cache lost)
                original_arr = cur
            self._pending_sig = ai_signature(opts)
            self._pending_arr = original_arr

        self.progress_bar.setValue(0)
        self.run_btn.setText("Cancel")

        self.worker = DeepSNRWorker(self.siril, opts, sequence_mode, arr=original_arr)
        self.worker.progress.connect(self._on_progress)
        self.worker.message.connect(self._on_message)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, frac):
        self.progress_bar.setValue(int(max(0.0, min(1.0, frac)) * 100))
        try:
            self.siril.update_progress("DeepSNR", max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def _on_message(self, text):
        self.log_view.appendPlainText(text)

    def _on_finished(self, ok, message):
        self.run_btn.setText("Run DeepSNR")
        self.run_btn.setEnabled(True)
        self._on_message(message)
        try:
            self.siril.reset_progress()
        except Exception:
            pass
        # Cache a successful single-image AI result so the modulation slider can
        # re-blend without re-running the model.
        payload = getattr(self.worker, "cache_payload", None) if self.worker else None
        if ok and payload is not None and self._pending_sig is not None:
            self._cache = {
                "ai_sig": self._pending_sig,
                "original_arr": self._pending_arr,
                "denoised_hwc": payload[1],
            }
            self._last_written_fp = getattr(self.worker, "written_fp", None)
        if ok:
            self.progress_bar.setValue(100)
        else:
            QMessageBox.warning(self, "DeepSNR", message)

    def closeEvent(self, event):
        try:
            self._collect_options()
            save_options(self.siril, self.opts)
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        try:
            self.siril.disconnect()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_arg_parser():
    def stride_arg(value):
        try:
            stride = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("stride must be an integer") from exc
        if stride < 2 or stride > 512 or stride % 2 != 0:
            raise argparse.ArgumentTypeError("stride must be an even number from 2 to 512")
        return stride

    def modulation_arg(value):
        try:
            modulation = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("modulation must be a number") from exc
        if not math.isfinite(modulation) or modulation < 0.0 or modulation > 1.0:
            raise argparse.ArgumentTypeError("modulation must be a finite number from 0.0 to 1.0")
        return modulation

    p = argparse.ArgumentParser(
        prog="DeepSNR.py",
        description="Denoise the loaded image or sequence using DeepSNR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exe", help="Optional last-resort path to a nonstandard "
                   "DeepSNR executable. Normally leave unset so PATH or the "
                   "standard CLI installer location is used.")
    p.add_argument("--weights", help="Optional path to a model weights file.")
    stretch = p.add_mutually_exclusive_group()
    stretch.add_argument("--linear", dest="linear", action="store_true",
                         help="Treat the image as linear and autostretch before denoising (default).")
    stretch.add_argument("--no-linear", dest="linear", action="store_false",
                         help="Treat the image as already stretched.")
    p.set_defaults(linear=None)
    p.add_argument("--stride", type=stride_arg, help="Custom tile stride (even, 2..512).")
    p.add_argument("--model", type=int, choices=(1, 2),
                   help="Model version: 1 (colour only) or 2 (colour and mono, default).")
    p.add_argument("--modulation", type=modulation_arg,
                   help="Denoise strength 0.0-1.0 (1.0 = fully denoised, default).")
    p.add_argument("--sequence", action="store_true",
                   help="Process the loaded FITS sequence instead of the loaded image.")
    return p


def run_cli(siril, argv):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    opts = load_options(siril)
    if args.exe:
        opts.exe = args.exe
    if args.weights is not None:
        opts.weights = args.weights
    if args.linear is not None:
        opts.linear = args.linear
    if args.stride is not None:
        opts.custom_stride = True
        opts.stride = args.stride
    if args.model is not None:
        opts.model = args.model
    if args.modulation is not None:
        opts.modulation = args.modulation

    status, message = detect_deepsnr_version(opts.exe)
    siril.log(f"DeepSNR: {message}", LogColor.GREEN if status == "ok" else LogColor.SALMON)
    if status != "ok":
        return 1
    notice = latest_cli_update_notice("deepsnr", parse_version_tuple(message))
    if notice:
        siril.log(f"DeepSNR: {notice}", LogColor.SALMON)

    def progress_cb(frac):
        try:
            siril.update_progress("DeepSNR", max(0.0, min(1.0, frac)))
        except Exception:
            pass

    def log_cb(text):
        siril.log(f"DeepSNR: {text}", LogColor.DEFAULT)

    try:
        if args.sequence:
            if not siril.is_sequence_loaded():
                siril.log("DeepSNR: no sequence is loaded.", LogColor.RED)
                return 1
            ok = process_sequence(siril, opts, progress_cb=progress_cb,
                                  log_cb=log_cb)
        else:
            if not siril.is_image_loaded():
                siril.log("DeepSNR: no image is loaded.", LogColor.RED)
                return 1
            ok = process_single_image(siril, opts, progress_cb=progress_cb,
                                      log_cb=log_cb)
    except DeepSNRError as exc:
        siril.log(f"DeepSNR: {exc}", LogColor.RED)
        return 1
    finally:
        try:
            siril.reset_progress()
        except Exception:
            pass
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli_argv = [a for a in sys.argv[1:] if a not in ("--", )]

    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        print("DeepSNR: could not connect to Siril.", file=sys.stderr)
        sys.exit(1)

    try:
        siril.cmd("requires", "1.4.0")
    except Exception as exc:
        siril.log(f"DeepSNR: {exc}", LogColor.RED)
        siril.disconnect()
        sys.exit(1)

    if cli_argv:
        try:
            rc = run_cli(siril, cli_argv)
        finally:
            siril.disconnect()
        sys.exit(rc)

    app = QApplication.instance() or QApplication(sys.argv)
    gui = DeepSNRGUI(siril)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
