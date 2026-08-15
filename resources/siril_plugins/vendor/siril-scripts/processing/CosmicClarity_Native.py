"""
# =============================================================================
# Cosmic Clarity AI4 for Siril
# =============================================================================
AI-powered sharpening, denoising, super resolution and star removal
for astrophotography images, running directly inside Siril.

Based on the Cosmic Clarity AI engines from SetiAstro Suite Pro.
Original engine code and models © SetiAstro.

Siril adaptation © Adrian Knagg-Baugh 2026.

Script version: 1.2.0
=====================
1.0.0: Original release
1.0.1: Update to use a mirror of the required model files at huggingface.co
       (the original google drive link got absolutely hammered and it's not fair
       on SetiAstro to hammer his file to the point of unavailability)
1.1.0: Add support for new AI4 versions of Darkstar, Super-resolution and
       Satellite Removal functions and add support for onnxruntime to improve
       acceleration for AMD GPU users on Windows.
       Add persistent settings, visual status of which models are available and
       the ability to set the models directory so you can use an existing
       SetiAstroSuite Pro installation models directory.
1.1.1: Fix some bugs in the new models. Everything works now, however there may
       still be an issue that the onnxruntime models are not running as fast as
       they should. This is still under investigation.
1.2.0: Add Walking Noise denoising model variant (deep_denoise_*_AI4_1w).
       Model variant is now selected via a combo box (Full / Lite / Walking Noise)
       in the Denoise tab, and via --walking on the CLI.
       Fix a bug in handling tile padding that affected ONNX models with certain
       image sizes.

SPDX-License-Identifier: GPL-3.0-or-later

This script provides the ability to inference the Cosmic Clarity models without
requiring a SetiAstroSuite Pro installation plus an additional python venv
(separate to the Siril one and required to be python-3.12) for SetiAstroSuite
which also has all the GPU libraries installed. This may save over a gigabyte of
storage.

NOTE: Although this script is based on original inferencing code written for
SetiAstroSuite Pro, it is not written by the author of that software. Bugs
MUST be reported to https://gitlab.com/free-astro/siril-scripts, not to
SetiAstro.

Usage (GUI):  Run from Siril's Script menu with an image loaded.
Usage (CLI):  pyscript CosmicClarity_Native.py --mode sharpen --stellar-amount 0.6
              pyscript CosmicClarity_Native --help
=============================================================================
"""

import sirilpy as s

# ---------------------------------------------------------------------------
# Runtime package bootstrap  (must come before any other imports)
# ---------------------------------------------------------------------------
s.ensure_installed("PyQt6", "requests", "astropy")

try:
    s.ensure_installed("sep")
except Exception:
    pass

_th = s.TorchHelper()
_th.ensure_torch()

# torchvision is normally installed alongside PyTorch by ensure_torch() or the
# GPU_Manager.py script.
# It is only required for satellite trail removal, where it provides the
# ResNet18 and MobileNetV2 backbone architectures used by the two trail-detection
# classifiers (BinaryClassificationCNN / BinaryClassificationCNN2).
# The NAFNet removal network (SatelliteRemoverCNN) does not need it.
import importlib.util as _ilu
_HAS_TORCHVISION = _ilu.find_spec("torchvision") is not None

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os, sys, math, time, shutil, zipfile, tempfile, traceback, platform
import argparse, atexit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from astropy.io import fits

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QComboBox, QGroupBox,
    QTabWidget, QProgressBar, QFrame, QMessageBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QDialog, QTextBrowser, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

try:
    import sep as _sep
    _HAS_SEP = True
except ImportError:
    _HAS_SEP = False


# =============================================================================
# Constants
# =============================================================================

VERSION        = "1.2.0"

MODEL_FILES: dict[str, str] = {
    "sharpen_stellar":         "deep_sharp_stellar_AI4.pth",
    "sharpen_stellar_onnx":    "deep_sharp_stellar_AI4.onnx",
    "sharpen_nonstellar":      "deep_nonstellar_sharp_conditional_psf_AI4.pth",
    "sharpen_nonstellar_onnx": "deep_nonstellar_sharp_conditional_psf_AI4.onnx",
    "denoise_mono":            "deep_denoise_mono_AI4.pth",
    "denoise_mono_onnx":       "deep_denoise_mono_AI4.onnx",
    "denoise_color":           "deep_denoise_color_AI4.pth",
    "denoise_color_onnx":      "deep_denoise_color_AI4.onnx",
    "denoise_mono_lite":       "deep_denoise_mono_AI4_lite.pth",
    "denoise_mono_lite_onnx":  "deep_denoise_mono_AI4_lite.onnx",
    "denoise_color_lite":      "deep_denoise_color_AI4_lite.pth",
    "denoise_color_lite_onnx": "deep_denoise_color_AI4_lite.onnx",
    "denoise_mono_1w":         "deep_denoise_mono_AI4_1w.pth",
    "denoise_mono_1w_onnx":    "deep_denoise_mono_AI4_1w.onnx",
    "denoise_color_1w":        "deep_denoise_color_AI4_1w.pth",
    "denoise_color_1w_onnx":   "deep_denoise_color_AI4_1w.onnx",
    "superres_2x":             "superres_2x.pth",
    "superres_2x_onnx":        "superres_2x.onnx",
    "superres_3x":             "superres_3x.pth",
    "superres_3x_onnx":        "superres_3x.onnx",
    "superres_4x":             "superres_4x.pth",
    "superres_4x_onnx":        "superres_4x.onnx",
    "darkstar_mono":           "darkstar_mono_AI4.pt",    # was darkstar_v2.1.pth
    "darkstar_mono_onnx":      "darkstar_mono_AI4.onnx",
    "darkstar_color":          "darkstar_color_AI4.pt",   # was darkstar_v2.1c.pth
    "darkstar_color_onnx":     "darkstar_color_AI4.onnx",
    "sat_detect1":             "satellite_trail_detector_AI3.5.pth",
    "sat_detect2":             "satellite_trail_detector_mobilenetv2.5.pth",
    "sat_remove":              "satelliteRemovalAI4.pth", # was satelliteremovalAI3.5.pth
}

MODE_MODELS: dict[str, list[str]] = {
    "sharpen":      ["sharpen_stellar", "sharpen_nonstellar"],
    "denoise":      ["denoise_mono",     "denoise_color"],
    "denoise_lite": ["denoise_mono_lite", "denoise_color_lite"],
    "denoise_1w":   ["denoise_mono_1w",  "denoise_color_1w"],
    "superres":     ["superres_2x", "superres_3x", "superres_4x"],
    "darkstar":     ["darkstar_mono", "darkstar_color"],
    "satellite":    ["sat_detect1", "sat_detect2", "sat_remove"],
}

# =============================================================================
# Siril Interface
# =============================================================================

_SIRIL_IFACE: Optional[Any] = None

def set_siril_iface(iface) -> None:
    global _SIRIL_IFACE
    _SIRIL_IFACE = iface

# =============================================================================
# Models directory
# =============================================================================

_MODELS_DIR: Optional[Path] = None
_MODELS_DIR_IS_OVERRIDE: bool = False

def set_models_dir_override(path: Optional[Path]) -> None:
    """
    Point the script at an existing models directory (e.g. a SASpro installation).
    Pass None to revert to the default Siril user-data location.
    Flushes the model cache so the next load picks up files from the new path.
    """
    global _MODELS_DIR, _MODELS_DIR_IS_OVERRIDE
    if path is None:
        _MODELS_DIR = None          # models_dir() will recompute the default
        _MODELS_DIR_IS_OVERRIDE = False
    else:
        _MODELS_DIR = Path(path)
        _MODELS_DIR_IS_OVERRIDE = True
    _MODEL_CACHE.clear()


def models_dir_override() -> Optional[Path]:
    """Return the active override path, or None if using the default."""
    return _MODELS_DIR if _MODELS_DIR_IS_OVERRIDE else None


def init_models_dir(siril_iface) -> Path:
    global _MODELS_DIR
    _MODELS_DIR = Path(siril_iface.get_siril_userdatadir()) / "cosmic_clarity"
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _MODELS_DIR


def models_dir() -> Path:
    global _MODELS_DIR
    if _MODELS_DIR is not None:
        return _MODELS_DIR
    try:
        _si = s.SirilInterface()
        _si.connect()
        base = Path(_si.get_siril_userdatadir())
        _si.disconnect()
    except Exception:
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA",
                         str(Path.home() / "AppData" / "Local"))) / "siril"
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support" / "siril"
        else:
            xdg  = os.environ.get("XDG_DATA_HOME",
                                   str(Path.home() / ".local" / "share"))
            base = Path(xdg) / "siril"
    _MODELS_DIR = base / "cosmic_clarity"
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return _MODELS_DIR


def model_path(key: str) -> Path:
    return models_dir() / MODEL_FILES[key]


def models_installed(mode: str) -> bool:
    return all(model_path(k).exists() for k in MODE_MODELS.get(mode, []))


def any_models_installed() -> bool:
    return any(model_path(k).exists() for k in MODEL_FILES)


# =============================================================================
# Model download and installation
# =============================================================================

def _looks_like_html(data: bytes) -> bool:
    head = (data or b"").lstrip()[:256].lower()
    return head.startswith(b"<!doctype html") or b"<html" in head


def _parse_gdrive_confirm_form(html: str):
    import re
    m = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if not m:
        return None, None
    action = m.group(1)
    params: dict[str, str] = {}
    for name, val in re.findall(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html
    ):
        params[name] = val
    return action, params


HF_MODELS_URL = (
    "https://huggingface.co/ajekb78/cc_models_mirror/resolve/main/models.zip"
)


def download_models(
    dst: Path,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
    cancel_cb:   Optional[Callable[[], bool]]    = None,
) -> Path:
    """Download the Cosmic Clarity models zip from HuggingFace."""
    import requests

    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    log("Connecting to HuggingFace…")
    with requests.Session() as sess:
        r = sess.get(HF_MODELS_URL, stream=True, timeout=60,
                     allow_redirects=True)
        r.raise_for_status()

        total  = int(r.headers.get("Content-Length") or 0)
        done   = 0
        t_last = time.time()

        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if cancel_cb and cancel_cb():
                    fh.close()
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("Download cancelled.")
                if not chunk:
                    continue
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - t_last >= 0.5:
                    t_last = now
                    if total > 0:
                        log(f"Downloading… {done * 100 // total}%  "
                            f"({done // (1024*1024)} / {total // (1024*1024)} MB)")
                    else:
                        log(f"Downloading… {done // (1024*1024)} MB received")

    os.replace(str(tmp), str(dst))
    log("Download complete.")
    return dst

def install_models_zip(
    zip_path: Path,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    dst    = models_dir()
    pid    = os.getpid()
    tmp_ex = Path(tempfile.gettempdir()) / f"cc_siril_extract_{pid}"
    tmp_st = Path(tempfile.gettempdir()) / f"cc_siril_stage_{pid}"

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    shutil.rmtree(tmp_ex, ignore_errors=True)
    shutil.rmtree(tmp_st, ignore_errors=True)

    try:
        log("Extracting archive…")
        tmp_ex.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(tmp_ex)

        root = tmp_ex
        kids = list(root.iterdir())
        if len(kids) == 1 and kids[0].is_dir():
            root = kids[0]

        if not any(p.suffix.lower() in (".pth", ".pt", ".onnx") for p in root.rglob("*")):
            raise RuntimeError(
                "The archive contains no model files. "
                "It may be corrupt or from the wrong source."
            )

        log(f"Installing models to: {dst}")
        shutil.copytree(root, tmp_st)

        dst.mkdir(parents=True, exist_ok=True)
        for item in dst.iterdir():
            try:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            except Exception:
                pass

        for item in tmp_st.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        log("Models installed successfully.")
    finally:
        shutil.rmtree(tmp_ex, ignore_errors=True)
        shutil.rmtree(tmp_st, ignore_errors=True)

def _safe_prune_list_onnx() -> tuple[list[Path], int]:
    """
    Return (removable, skipped) where removable are ONNX paths that have a
    Torch counterpart present, and skipped is the count that were omitted
    because no counterpart exists.
    """
    removable: list[Path] = []
    skipped = 0
    seen: set[Path] = set()
    for rows in _MODE_MODEL_INFO.values():
        for _, pth_key, onnx_key in rows:
            if not onnx_key or onnx_key not in MODEL_FILES:
                continue
            onnx_path = model_path(onnx_key)
            if onnx_path in seen or not onnx_path.exists():
                continue
            seen.add(onnx_path)
            pth_path = model_path(pth_key) if pth_key in MODEL_FILES else None
            if pth_path and pth_path.exists():
                removable.append(onnx_path)
            else:
                skipped += 1
    return removable, skipped


def _safe_prune_list_torch() -> tuple[list[Path], int]:
    """
    Return (removable, skipped) where removable are Torch paths that have an
    ONNX counterpart present, and skipped is the count that were omitted
    because no counterpart exists.
    """
    removable: list[Path] = []
    skipped = 0
    seen: set[Path] = set()
    for rows in _MODE_MODEL_INFO.values():
        for _, pth_key, onnx_key in rows:
            if pth_key not in MODEL_FILES:
                continue
            pth_path = model_path(pth_key)
            if pth_path in seen or not pth_path.exists():
                continue
            seen.add(pth_path)
            onnx_path = (model_path(onnx_key)
                         if onnx_key and onnx_key in MODEL_FILES else None)
            if onnx_path and onnx_path.exists():
                removable.append(pth_path)
            else:
                skipped += 1
    return removable, skipped


def _delete_files(
    files: list[Path],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    removed = 0
    for f in files:
        try:
            f.unlink()
            removed += 1
            if progress_cb:
                progress_cb(f"Removed: {f.name}")
        except Exception as exc:
            if progress_cb:
                progress_cb(f"Error removing {f.name}: {exc}")
    return removed

def _orphaned_model_files(extension: str) -> list[Path]:
    """
    Return all files in the models directory with the given extension that are
    not referenced by any entry in MODEL_FILES (i.e. leftover from old versions).
    """
    known = {
        model_path(k).resolve()
        for k, v in MODEL_FILES.items()
        if v.endswith(extension)
    }
    return [
        f for f in models_dir().rglob(f"*{extension}")
        if f.resolve() not in known
    ]

def remove_onnx_models(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    files, skipped = _safe_prune_list_onnx()
    orphans = _orphaned_model_files(".onnx")
    all_files = files + [f for f in orphans if f not in files]
    removed = _delete_files(all_files, progress_cb)
    if progress_cb:
        msg = f"Removed {removed} ONNX file(s)."
        if orphans:
            msg += f"  ({len(orphans)} obsolete file(s) included.)"
        if skipped:
            msg += (f"  {skipped} current file(s) skipped — "
                    "no Torch counterpart present.")
        progress_cb(msg)
    return removed


def remove_torch_models(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    files, skipped = _safe_prune_list_torch()
    orphans = (
        _orphaned_model_files(".pth")
        + _orphaned_model_files(".pt")
    )
    all_files = files + [f for f in orphans if f not in files]
    removed = _delete_files(all_files, progress_cb)
    if progress_cb:
        msg = f"Removed {removed} Torch file(s)."
        if orphans:
            msg += f"  ({len(orphans)} obsolete file(s) included.)"
        if skipped:
            msg += (f"  {skipped} current file(s) skipped — "
                    "no ONNX counterpart present.")
        progress_cb(msg)
    return removed

def remove_obsolete_models(
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    orphans = (
        _orphaned_model_files(".pth")
        + _orphaned_model_files(".pt")
        + _orphaned_model_files(".onnx")
    )
    removed = _delete_files(orphans, progress_cb)
    if progress_cb:
        progress_cb(
            f"Removed {removed} obsolete file(s)."
            if removed else "No obsolete model files found."
        )
    return removed

# =============================================================================
# Device selection
# =============================================================================

def get_device(use_gpu: bool = True) -> torch.device:
    if not use_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available() and platform.machine() == "arm64":
            return torch.device("mps")
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except Exception:
        pass
    try:
        import torch_directml  # type: ignore
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def _autocast(device: torch.device):
    """Return an autocast context appropriate for the device, or a no-op."""
    try:
        if device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                return torch.amp.autocast(device_type="cuda")
        elif device.type == "mps":
            return torch.amp.autocast(device_type="mps")
    except Exception:
        pass
    return nullcontext()

# =============================================================================
# ONNX Runtime helpers
# =============================================================================

def _get_ort():
    """Import onnxruntime if available, else return None."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        return None

def _ort_providers(use_gpu: bool) -> list[str]:
    """Return an ordered provider list for ORT session creation."""
    # Prefer asking Siril directly — it knows which ORT EPs are properly
    # installed for the current platform and GPU configuration.
    if _SIRIL_IFACE is not None:
        try:
            oh = _SIRIL_IFACE.ONNXHelper()
            return oh.get_execution_providers_ordered(ai_gpu_acceleration=use_gpu)
        except Exception:
            pass  # fall through to the manual list below

    # Fallback: build the provider list ourselves if the interface is unavailable
    # (e.g. during unit tests or very early model-status queries).
    ort = _get_ort()
    if ort is None:
        return []
    available = ort.get_available_providers()
    prefs = []
    if use_gpu:
        if "CUDAExecutionProvider" in available:
            prefs.append("CUDAExecutionProvider")
        if "CoreMLExecutionProvider" in available:
            prefs.append("CoreMLExecutionProvider")
        if "DmlExecutionProvider" in available:
            prefs.append("DmlExecutionProvider")
    prefs.append("CPUExecutionProvider")
    return [p for p in prefs if p in available]

def _ort_session(path: Path, use_gpu: bool = False):
    ort = _get_ort()
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")

    providers       = _ort_providers(use_gpu)
    provider_options: list[dict] = []

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_mem_pattern       = True
    so.enable_cpu_mem_arena     = True
    if providers and providers[0] == "DmlExecutionProvider":
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
    else:
        so.intra_op_num_threads = 0   # all cores fine for CUDA / CPU
        so.inter_op_num_threads = 0

    if providers and providers[0] == "CUDAExecutionProvider":
        cuda_opts = {
            "device_id":                    "0",
            "arena_extend_strategy":        "kNextPowerOfTwo",
            "cudnn_conv_algo_search":       "HEURISTIC",
            "do_copy_in_default_stream":    "0",
            "cudnn_conv_use_max_workspace": "1",
            "enable_cuda_graph":            "0",
        }
        for p in providers:
            provider_options.append(cuda_opts if p == "CUDAExecutionProvider" else {})

    elif providers and providers[0] in ("CoreMLExecutionProvider",
                                        "DmlExecutionProvider"):
        provider_options = [{} for _ in providers]

    else:
        provider_options = [{} for _ in providers]

    sess = ort.InferenceSession(str(path), sess_options=so,
                                providers=providers,
                                provider_options=provider_options)

    # Warm-up: run one dummy inference so cuDNN benchmarks and caches
    # the convolution algorithms for the most common tile shape (512×512).
    # Without this the first real tile pays the benchmark cost and shows
    # as a spike in GPU utilisation.
    if providers and providers[0] == "CUDAExecutionProvider":
        try:
            dummy_h, dummy_w = 512, 512
            for inp in sess.get_inputs():
                shp = inp.shape
                # Build a shape where dynamic dims use the dummy size
                concrete = [
                    (dummy_h if (d is None or str(d) in ("h", "H", "height", "?")) else
                     dummy_w if (d is None or str(d) in ("w", "W", "width"))  else
                     int(d))
                    for d in shp
                ]
                dummy = {inp.name: np.zeros(concrete, dtype=np.float32)}
            sess.run(None, dummy)
        except Exception:
            pass   # warm-up is best-effort

    return sess

def _ort_pick_io_names(session) -> tuple[str, Optional[str], str]:
    """
    Robustly identify (img_input_name, psf_input_name_or_None, output_name)
    from an ORT session by inspecting input ranks.
    """
    ins = session.get_inputs()
    out_name = session.get_outputs()[0].name
    img_name = psf_name = None
    for i in ins:
        rank = len(i.shape) if i.shape else 0
        if rank == 4:
            img_name = i.name
        elif rank in (1, 2):
            psf_name = i.name
    if img_name is None:
        img_name = ins[0].name
    if len(ins) == 1:
        psf_name = None
    elif psf_name is None and len(ins) > 1:
        psf_name = ins[1].name
    return img_name, psf_name, out_name

# =============================================================================
# Model architectures  (must exactly match the training definitions)
# =============================================================================

# --- Shared NAFNet building blocks -------------------------------------------

class _LN2d(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu  = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + self.eps).sqrt() * self.weight + self.bias


class _SG(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class _NAFBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1  = _LN2d(ch)
        self.conv1  = nn.Conv2d(ch, ch*2, 1, bias=True)
        self.dwconv = nn.Conv2d(ch*2, ch*2, 3, padding=1, groups=ch*2, bias=True)
        self.sg     = _SG()
        self.sca    = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch, 1, bias=True))
        self.conv2  = nn.Conv2d(ch, ch, 1, bias=True)
        self.norm2  = _LN2d(ch)
        self.ffn1   = nn.Conv2d(ch, ch*2, 1, bias=True)
        self.ffn2   = nn.Conv2d(ch, ch, 1, bias=True)
        self.beta   = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.gamma  = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sg(self.dwconv(self.conv1(self.norm1(x))))
        y = y * self.sca(y)
        x = x + self.conv2(y) * self.beta
        y = self.sg(self.ffn1(self.norm2(x)))
        return x + self.ffn2(y) * self.gamma


def _nafnet_body(width: int, enc_nums: tuple, dec_nums: tuple, mid_num: int) -> tuple:
    encoders, downs, decoders, ups = (
        nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
    )
    ch = width
    for n in enc_nums:
        encoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))
        downs.append(nn.Conv2d(ch, ch*2, 2, stride=2, bias=True))
        ch *= 2
    mid = nn.Sequential(*[_NAFBlock(ch) for _ in range(mid_num)])
    for n in dec_nums:
        ups.append(nn.Sequential(nn.Conv2d(ch, ch*2, 1, bias=True), nn.PixelShuffle(2)))
        ch //= 2
        decoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))
    return encoders, downs, mid, decoders, ups


# --- Sharpening: stellar (RGB → RGB) -----------------------------------------

class NAFNetSharpen(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=32,
                 enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                 middle_blk_num=4, residual_out=True, clamp_out=False):
        super().__init__()
        self.intro   = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending  = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
        self.encoders, self.downs, self.middle, self.decoders, self.ups = \
            _nafnet_body(width, enc_blk_nums, dec_blk_nums, middle_blk_num)
        self.residual_out = residual_out
        self.clamp_out    = clamp_out

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intro(x)
        sk = []
        for e, d in zip(self.encoders, self.downs):
            x = e(x); sk.append(x); x = d(x)
        x = self.middle(x)
        for u, d in zip(self.ups, self.decoders):
            x = u(x); x = x + sk.pop(); x = d(x)
        return self.ending(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self._delta(x) if self.residual_out else self._delta(x)
        return y.clamp(0, 1) if self.clamp_out else y


# --- Sharpening: non-stellar PSF-conditional (RGB + PSF → RGB) ---------------

class NAFNetSharpenPSF(NAFNetSharpen):
    def __init__(self, **kw):
        super().__init__(in_ch=4, out_ch=3, **kw)

    def forward(self, x_rgb: torch.Tensor, psf_t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, _, h, w = x_rgb.shape
        psf_map = psf_t.view(b, 1, 1, 1).expand(b, 1, h, w)
        x4 = torch.cat([x_rgb, psf_map], dim=1)
        y = x_rgb + self._delta(x4) if self.residual_out else self._delta(x4)
        return y.clamp(0, 1) if self.clamp_out else y


# --- Denoising: NAFNet (full width=64, lite width=32) ------------------------

class NAFNetDenoise(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, width=32,
                 enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                 middle_blk_num=4, residual_out=True, clamp_out=False):
        super().__init__()
        self.intro   = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending  = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
        self.encoders, self.downs, self.middle, self.decoders, self.ups = \
            _nafnet_body(width, enc_blk_nums, dec_blk_nums, middle_blk_num)
        self.residual_out = residual_out
        self.clamp_out    = clamp_out

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intro(x)
        sk = []
        for e, d in zip(self.encoders, self.downs):
            x = e(x); sk.append(x); x = d(x)
        x = self.middle(x)
        for u, d in zip(self.ups, self.decoders):
            x = u(x); x = x + sk.pop(); x = d(x)
        return self.ending(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self._delta(x) if self.residual_out else self._delta(x)
        return y.clamp(0, 1) if self.clamp_out else y


# --- Super resolution: U-Net CNN (per-channel) --------------------------------

class _RB(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        R, rl = _RB, lambda: nn.ReLU(True)
        self.encoder1 = nn.Sequential(nn.Conv2d(3,16,3,padding=1), rl(), R(16))
        self.encoder2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), rl(), R(32))
        self.encoder3 = nn.Sequential(nn.Conv2d(32,64,3,padding=2,dilation=2), rl(), R(64))
        self.encoder4 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), rl(), R(128))
        self.encoder5 = nn.Sequential(nn.Conv2d(128,256,3,padding=2,dilation=2), rl(), R(256))
        self.decoder5 = nn.Sequential(nn.Conv2d(256+128,128,3,padding=1), rl(), R(128))
        self.decoder4 = nn.Sequential(nn.Conv2d(128+64,64,3,padding=1), rl(), R(64))
        self.decoder3 = nn.Sequential(nn.Conv2d(64+32,32,3,padding=1), rl(), R(32))
        self.decoder2 = nn.Sequential(nn.Conv2d(32+16,16,3,padding=1), rl(), R(16))
        self.decoder1 = nn.Sequential(nn.Conv2d(16,3,3,padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1=self.encoder1(x); e2=self.encoder2(e1); e3=self.encoder3(e2)
        e4=self.encoder4(e3); e5=self.encoder5(e4)
        d=self.decoder5(torch.cat([e5,e4],1)); d=self.decoder4(torch.cat([d,e3],1))
        d=self.decoder3(torch.cat([d,e2],1)); d=self.decoder2(torch.cat([d,e1],1))
        return self.decoder1(d)


# --- Star removal: DarkStar CNN ----------------------------------------------

class DarkStarNAFNet(nn.Module):
    """NAFNet-based star removal for AI4 DarkStar checkpoints."""
    def __init__(self, in_ch: int = 3, out_ch: int = 3, width: int = 32,
                 enc_blk_nums=(2, 4, 6, 8), dec_blk_nums=(2, 2, 2, 2),
                 middle_blk_num: int = 4):
        super().__init__()
        self.padder_size = 2 ** len(enc_blk_nums)
        self.intro   = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending  = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2
        self.middle = nn.Sequential(*[_NAFBlock(ch) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=True),
                                          nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        # Pad to multiple of padder_size
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        x = self.intro(x)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x); encs.append(x); x = down(x)
        x = self.middle(x)
        for up, dec in zip(self.ups, self.decoders):
            x = up(x); x = x + encs.pop(); x = dec(x)
        x = self.ending(x)
        if ph or pw:
            x = x[:, :, :h, :w]
        return torch.clamp(x + inp, 0.0, 1.0)

# --- Satellite removal: NAFNet remover + dual ResNet/MobileNet classifiers ----

class NAFNetSatelliteRemover(nn.Module):
    """NAFNet-based satellite trail remover for AI4 checkpoint (residual output)."""
    def __init__(self, width: int = 32,
                 enc_blk_nums=(2, 4, 6, 8), dec_blk_nums=(2, 2, 2, 2),
                 middle_blk_num: int = 4, residual_out: bool = True):
        super().__init__()
        self.residual_out = residual_out
        self.intro   = nn.Conv2d(3, width, 3, padding=1, bias=True)
        self.ending  = nn.Conv2d(width, 3, 3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
            ch *= 2
        self.middle = nn.Sequential(*[_NAFBlock(ch) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=True),
                                          nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[_NAFBlock(ch) for _ in range(n)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        y = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y); skips.append(y); y = down(y)
        y = self.middle(y)
        for up, dec in zip(self.ups, self.decoders):
            y = up(y); y = y + skips.pop(); y = dec(y)
        delta = self.ending(y)
        return (x0 + delta) if self.residual_out else delta

class BinaryClassificationCNN(nn.Module):
    """ResNet18-backbone classifier — detector 1."""
    def __init__(self):
        super().__init__()
        from torchvision import models as _tv_models
        self.pre_conv1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.pre_conv2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64), nn.ReLU())
        self.features  = _tv_models.resnet18(weights=None)
        self.features.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features.fc    = nn.Linear(self.features.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.pre_conv2(self.pre_conv1(x)))


class BinaryClassificationCNN2(nn.Module):
    """MobileNetV2-backbone classifier — detector 2."""
    def __init__(self):
        super().__init__()
        from torchvision import models as _tv_models
        self.pre_conv1  = nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.pre_conv2  = nn.Sequential(nn.Conv2d(32,64,3,padding=1,bias=False),nn.BatchNorm2d(64), nn.ReLU())
        self.mobilenet  = _tv_models.mobilenet_v2(weights=None)
        self.mobilenet.features[0][0] = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_f = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_f, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mobilenet(self.pre_conv2(self.pre_conv1(x)))

_MODEL_CACHE: dict[str, Any] = {}

_BACKEND_PREF: str = "auto"  # "auto", "torch", "onnx"

def set_backend_pref(pref: str) -> None:
    """Update the global backend preference and flush the model cache."""
    global _BACKEND_PREF
    assert pref in ("auto", "torch", "onnx")
    _BACKEND_PREF = pref
    _MODEL_CACHE.clear()

def _should_use_onnx(device: "torch.device", onnx_file_ok: bool) -> bool:
    """
    Resolve whether to use ONNX Runtime for a given model, taking the user's
    preference into account. Falls back gracefully when files are absent.
    """
    ort_ok = _get_ort() is not None
    if not ort_ok or not onnx_file_ok:
        return False
    if _BACKEND_PREF == "onnx":
        return True
    if _BACKEND_PREF == "torch":
        return False
    # "auto": prefer ONNX only when not on a native GPU accelerator
    try:
        return device.type not in ("cuda", "mps", "xpu")
    except Exception:
        return True

_SHARPEN_CFG      = dict(width=32, enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                         middle_blk_num=4, residual_out=True, clamp_out=False)
_DENOISE_CFG_FULL = dict(width=64, enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                         middle_blk_num=4, residual_out=True, clamp_out=False)
_DENOISE_CFG_LITE = dict(width=32, enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                         middle_blk_num=4, residual_out=True, clamp_out=False)


@dataclass
class SharpenModels:
    device:  Any          # torch.device or "onnx"
    is_onnx: bool
    stellar: Any          # nn.Module or ort.InferenceSession
    ns_cond: Any


@dataclass
class DenoiseModels:
    device:  Any
    is_onnx: bool
    mono:    Any
    color:   Any
    variant: str = "full"


@dataclass
class SuperresModel:
    device:  Any
    is_onnx: bool
    model:   Any
    scale:   int


@dataclass
class DarkstarModels:
    device:  Any
    is_onnx: bool
    model:   Any          # mono or color, selected at load time


@dataclass
class SatelliteModels:
    device:   Any
    is_onnx:  bool        # only the remover may use ORT (detectors need torchvision)
    detect1:  Any
    detect2:  Any
    remover:  Any

def _load_state(net: nn.Module, path: Path, device: torch.device) -> nn.Module:
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        # Try all known checkpoint wrapper keys in priority order
        for key in ("model_state_dict", "state_dict", "model_state",
                    "model", "net", "network", "params_ema", "params"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
        else:
            # DarkStar stage1. prefix (legacy checkpoints)
            if any(k.startswith("stage1.") for k in sd):
                sd = {k[len("stage1."):]: v for k, v in sd.items()
                      if k.startswith("stage1.")}
    # Strip DataParallel / torch.compile wrappers
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
          for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net.to(device)

def load_sharpen_models(use_gpu: bool = True) -> SharpenModels:
    key = f"sharpen_{use_gpu}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = get_device(use_gpu)

    # Prefer ONNX when not on a native GPU (CUDA/MPS) — avoids a full torch graph
    _ort = _get_ort()
    use_onnx = _should_use_onnx(device,
                model_path("sharpen_stellar_onnx").exists()
                and model_path("sharpen_nonstellar_onnx").exists())

    if use_onnx:
        providers = _ort_providers(use_gpu)
        stellar = _ort_session(model_path("sharpen_stellar_onnx"), use_gpu)
        ns_cond = _ort_session(model_path("sharpen_nonstellar_onnx"), use_gpu)
        m = SharpenModels(device="onnx", is_onnx=True,
                          stellar=stellar, ns_cond=ns_cond)
    else:
        stellar = _load_state(NAFNetSharpen(**_SHARPEN_CFG),
                              model_path("sharpen_stellar"), device)
        ns_cond = _load_state(NAFNetSharpenPSF(**_SHARPEN_CFG),
                              model_path("sharpen_nonstellar"), device)
        m = SharpenModels(device=device, is_onnx=False,
                          stellar=stellar, ns_cond=ns_cond)

    _MODEL_CACHE[key] = m
    return m

def load_denoise_models(use_gpu: bool = True, lite: bool = False,
                        walking: bool = False) -> DenoiseModels:
    if walking:
        variant_str, suffix, cfg = "1w",   "_1w",   _DENOISE_CFG_FULL
    elif lite:
        variant_str, suffix, cfg = "lite", "_lite", _DENOISE_CFG_LITE
    else:
        variant_str, suffix, cfg = "full", "",      _DENOISE_CFG_FULL

    key = f"denoise_{use_gpu}_{variant_str}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = get_device(use_gpu)

    _ort = _get_ort()
    use_onnx = _should_use_onnx(device,
               model_path(f"denoise_mono{suffix}_onnx").exists()
               and model_path(f"denoise_color{suffix}_onnx").exists())

    if use_onnx:
        mono  = _ort_session(model_path(f"denoise_mono{suffix}_onnx"),  use_gpu)
        color = _ort_session(model_path(f"denoise_color{suffix}_onnx"), use_gpu)
        m = DenoiseModels(device="onnx", is_onnx=True, mono=mono, color=color,
                          variant=variant_str)
    else:
        mono  = _load_state(NAFNetDenoise(**cfg), model_path(f"denoise_mono{suffix}"),  device)
        color = _load_state(NAFNetDenoise(**cfg), model_path(f"denoise_color{suffix}"), device)
        m = DenoiseModels(device=device, is_onnx=False, mono=mono, color=color,
                          variant=variant_str)

    _MODEL_CACHE[key] = m
    return m

def load_superres_model(scale: int, use_gpu: bool = True) -> SuperresModel:
    key = f"superres_{scale}_{use_gpu}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = get_device(use_gpu)

    _ort = _get_ort()
    use_onnx = _should_use_onnx(device, model_path(f"superres_{scale}x_onnx").exists())

    if use_onnx:
        model = _ort_session(model_path(f"superres_{scale}x_onnx"), use_gpu)
        m = SuperresModel(device="onnx", is_onnx=True, model=model, scale=scale)
    else:
        model = _load_state(SuperResolutionCNN(), model_path(f"superres_{scale}x"), device)
        m = SuperresModel(device=device, is_onnx=False, model=model, scale=scale)

    _MODEL_CACHE[key] = m
    return m

_DARKSTAR_CFG = dict(width=32, enc_blk_nums=(2, 4, 6, 8),
                     dec_blk_nums=(2, 2, 2, 2), middle_blk_num=4)

def load_darkstar_models(use_gpu: bool = True, color: bool = True) -> DarkstarModels:
    key = f"darkstar_{use_gpu}_{color}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device   = get_device(use_gpu)
    pth_key  = "darkstar_color" if color else "darkstar_mono"
    onnx_key = "darkstar_color_onnx" if color else "darkstar_mono_onnx"

    _ort = _get_ort()
    use_onnx = _should_use_onnx(device, model_path(onnx_key).exists())

    if use_onnx:
        model = _ort_session(model_path(onnx_key), use_gpu)
        m = DarkstarModels(device="onnx", is_onnx=True, model=model)
    else:
        net = DarkStarNAFNet(**_DARKSTAR_CFG)
        model = _load_state(net, model_path(pth_key), device)
        m = DarkstarModels(device=device, is_onnx=False, model=model)

    _MODEL_CACHE[key] = m
    return m

def _load_state_lenient(net: nn.Module, path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model_state",
                    "model", "net", "network", "params_ema", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
          for k, v in ckpt.items()}
    msd = net.state_dict()
    filtered = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    net.load_state_dict(filtered, strict=False)
    net.eval()
    return net.to(device)

def load_satellite_models(use_gpu: bool = True) -> SatelliteModels:
    if not _HAS_TORCHVISION:
        raise RuntimeError(
            "torchvision is not available — required for satellite trail detection.")
    key = f"satellite_{use_gpu}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device  = get_device(use_gpu)
    detect1 = _load_state_lenient(BinaryClassificationCNN(),  model_path("sat_detect1"), device)
    detect2 = _load_state_lenient(BinaryClassificationCNN2(), model_path("sat_detect2"), device)

    # Remover: prefer ONNX on non-native-GPU backends
    _ort = _get_ort()
    use_onnx_rem = (_ort is not None
                    and device.type not in ("cuda", "mps", "xpu")
                    and model_path("sat_remove_onnx").exists()
                    if "sat_remove_onnx" in MODEL_FILES else False)

    if use_onnx_rem:
        remover = _ort_session(model_path("sat_remove_onnx"), use_gpu)
        m = SatelliteModels(device=device, is_onnx=True,
                            detect1=detect1, detect2=detect2, remover=remover)
    else:
        remover = _load_state_lenient(
            NAFNetSatelliteRemover(width=32, enc_blk_nums=(2,4,6,8),
                                   dec_blk_nums=(2,2,2,2), middle_blk_num=4),
            model_path("sat_remove"), device)
        m = SatelliteModels(device=device, is_onnx=False,
                            detect1=detect1, detect2=detect2, remover=remover)

    _MODEL_CACHE[key] = m
    return m

# =============================================================================
# Image utilities
# =============================================================================

# --- Padding -----------------------------------------------------------------

def _pad_mult16(arr2d: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad a 2-D array so H and W are multiples of 16 (NAFNet requirement)."""
    h, w = arr2d.shape
    ph, pw = (-h) % 16, (-w) % 16
    if ph == 0 and pw == 0:
        return arr2d, h, w
    return np.pad(arr2d, ((0, ph), (0, pw)), mode="reflect"), h, w


# --- Tiling ------------------------------------------------------------------

def _split_chunks(image: np.ndarray, chunk_size: int, overlap: int) -> list:
    H, W  = image.shape[:2]
    step  = max(1, chunk_size - overlap)
    out   = []
    for i in range(0, H, step):
        for j in range(0, W, step):
            ei = min(i + chunk_size, H)
            ej = min(j + chunk_size, W)
            if ei > i and ej > j:
                out.append((image[i:ei, j:ej], i, j))
    return out


def _stitch_border_ignore(
    chunks: list, shape2d: tuple, border: int = 16
) -> np.ndarray:
    """Stitch 2-D chunks, discarding border pixels to avoid seam artefacts."""
    H, W = shape2d
    out  = np.zeros((H, W), np.float32)
    wts  = np.zeros((H, W), np.float32)
    for chunk, i, j in chunks:
        h, w   = chunk.shape[:2]
        bh, bw = min(border, h // 2), min(border, w // 2)
        y0, y1 = i + bh, i + h - bh
        x0, x1 = j + bw, j + w - bw
        if y1 <= y0 or x1 <= x0:
            continue
        inner         = chunk[bh:h-bh, bw:w-bw]
        yy0, yy1      = max(0, y0), min(H, y1)
        xx0, xx1      = max(0, x0), min(W, x1)
        if yy1 <= yy0 or xx1 <= xx0:
            continue
        sy0 = yy0 - y0;  sy1 = sy0 + yy1 - yy0
        sx0 = xx0 - x0;  sx1 = sx0 + xx1 - xx0
        out[yy0:yy1, xx0:xx1] += inner[sy0:sy1, sx0:sx1]
        wts[yy0:yy1, xx0:xx1] += 1.0
    return out / np.maximum(wts, 1.0)


def _stitch_rgb_border_ignore(
    chunks: list, shape_hwc: tuple, border: int = 16
) -> np.ndarray:
    """Border-ignore stitching for HxWx3 RGB chunks."""
    H, W, C = shape_hwc
    out  = np.zeros((H, W, C), np.float32)
    wts  = np.zeros((H, W),    np.float32)
    for tile, i, j in chunks:
        h, w   = tile.shape[:2]
        bh, bw = min(border, h // 2), min(border, w // 2)
        y0, y1 = i + bh, i + h - bh
        x0, x1 = j + bw, j + w - bw
        if y1 <= y0 or x1 <= x0:
            continue
        inner         = tile[bh:h-bh, bw:w-bw]
        yy0, yy1      = max(0, y0), min(H, y1)
        xx0, xx1      = max(0, x0), min(W, x1)
        if yy1 <= yy0 or xx1 <= xx0:
            continue
        sy0 = yy0 - y0;  sy1 = sy0 + yy1 - yy0
        sx0 = xx0 - x0;  sx1 = sx0 + xx1 - xx0
        out[yy0:yy1, xx0:xx1, :] += inner[sy0:sy1, sx0:sx1, :]
        wts[yy0:yy1, xx0:xx1]    += 1.0
    return out / np.maximum(wts[:, :, None], 1.0)


def _stitch_rgb_soft_blend(
    chunks: list, shape_hwc: tuple, chunk_size: int, overlap: int, border: int = 5
) -> np.ndarray:
    """Soft-blend (linear ramp weight) stitching for DarkStar RGB tiles."""
    H, W, C = shape_hwc
    out  = np.zeros((H, W, C), np.float32)
    wsum = np.zeros((H, W),    np.float32)

    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        flat = np.ones(max(chunk_size - 2 * overlap, 1), dtype=np.float32)
        v    = np.concatenate([ramp, flat, ramp[::-1]])
        bw_f = np.outer(v, v).astype(np.float32)
    else:
        bw_f = np.ones((chunk_size, chunk_size), np.float32)

    for tile, i, j in chunks:
        th, tw = tile.shape[:2]
        top    = 0 if i == 0        else min(border, th // 2)
        left   = 0 if j == 0        else min(border, tw // 2)
        bot    = 0 if (i + th >= H) else min(border, th // 2)
        right  = 0 if (j + tw >= W) else min(border, tw // 2)
        inner  = tile[top:th-bot, left:tw-right, :]
        ih, iw = inner.shape[:2]
        rr0    = i + top
        cc0    = j + left
        bw     = bw_f[:ih, :iw]
        out [rr0:rr0+ih, cc0:cc0+iw, :] += inner * bw[:, :, None]
        wsum[rr0:rr0+ih, cc0:cc0+iw]    += bw
    return out / np.maximum(wsum[:, :, None], 1e-8)


# --- Border ------------------------------------------------------------------

def _add_border_2d(arr: np.ndarray, b: int = 16) -> np.ndarray:
    return np.pad(arr, ((b, b), (b, b)), mode="constant",
                  constant_values=float(np.median(arr)))


def _add_border_rgb(arr: np.ndarray, b: int = 16) -> np.ndarray:
    meds = np.median(arr, axis=(0, 1)).astype(np.float32)
    return np.stack(
        [np.pad(arr[..., c], ((b, b), (b, b)), mode="constant",
                constant_values=float(meds[c])) for c in range(3)],
        axis=-1,
    )


def _remove_border(arr: np.ndarray, b: int) -> np.ndarray:
    return arr[b:-b, b:-b] if arr.ndim == 2 else arr[b:-b, b:-b, :]


# --- Blending ----------------------------------------------------------------

def _blend(before: np.ndarray, after: np.ndarray, amount: float) -> np.ndarray:
    a = float(np.clip(amount, 0.0, 1.0))
    return (1.0 - a) * before + a * after


# --- YCbCr luminance / chroma ------------------------------------------------

_M_FWD = np.array([[ 0.299,     0.587,      0.114   ],
                   [-0.168736, -0.331264,    0.5     ],
                   [ 0.5,      -0.418688,   -0.081312]], np.float32)
_M_INV = np.array([[ 1.0,  0.0,       1.402    ],
                   [ 1.0, -0.344136, -0.714136  ],
                   [ 1.0,  1.772,     0.0       ]], np.float32)


def _extract_luminance(rgb: np.ndarray) -> tuple:
    ycbcr = rgb @ _M_FWD.T
    return ycbcr[..., 0], ycbcr[..., 1] + 0.5, ycbcr[..., 2] + 0.5


def _merge_luminance(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    ycbcr = np.stack([np.clip(y, 0, 1),
                      np.clip(cb, 0, 1) - 0.5,
                      np.clip(cr, 0, 1) - 0.5], axis=-1)
    return np.clip(ycbcr @ _M_INV.T, 0.0, 1.0).astype(np.float32, copy=False)


# --- Midtone-function stretch / unstretch ------------------------------------

def _stretch(image: np.ndarray, target_median: float = 0.25):
    x    = image.astype(np.float32, copy=True)
    omin = float(np.min(x))
    x   -= omin
    t    = float(target_median)

    if x.ndim == 2:
        m0 = float(np.median(x))
        om = [m0]
        if m0 > 0:
            denom = m0 * (t + x - 1.0) - t * x
            x = np.where(np.abs(denom) > 1e-12, (m0 - 1.0) * t * x / denom, x)
    else:
        om = []
        for c in range(3):
            m0 = float(np.median(x[..., c]))
            om.append(m0)
            if m0 > 0:
                denom = m0 * (t + x[..., c] - 1.0) - t * x[..., c]
                x[..., c] = np.where(np.abs(denom) > 1e-12,
                                      (m0 - 1.0) * t * x[..., c] / denom, x[..., c])
    return np.clip(x, 0, 1), omin, om


def _unstretch(image: np.ndarray, orig_meds: list, orig_min: float,
               target_median: float = 0.25) -> np.ndarray:
    y = image.astype(np.float32, copy=True)
    t = float(target_median)

    def _inv(yc: np.ndarray, m0: float) -> np.ndarray:
        denom = t * (m0 - 1.0 + yc) - yc * m0
        return np.where(np.abs(denom) > 1e-12, yc * m0 * (t - 1.0) / denom, yc)

    if y.ndim == 2:
        if float(orig_meds[0]) > 0:
            y = _inv(y, float(orig_meds[0]))
    else:
        for c in range(3):
            if float(orig_meds[c]) > 0:
                y[..., c] = _inv(y[..., c], float(orig_meds[c]))
    y += float(orig_min)
    return np.clip(y, 0, 1).astype(np.float32, copy=False)


# --- PSF utilities -----------------------------------------------------------

def _encode_psf(r: float) -> float:
    lo, hi = math.log2(1.0), math.log2(8.0)
    return float(np.clip((math.log2(max(r, 1.0)) - lo) / (hi - lo), 0.0, 1.0))


def _measure_psf(chunk2d: np.ndarray, default: float = 3.0) -> float:
    if not _HAS_SEP:
        return default
    try:
        import sep
        data = chunk2d.astype(np.float32, copy=False)
        bkg  = sep.Background(data)
        sub  = data - bkg.back()
        rms  = bkg.rms()
        if rms.size == 0:
            return default
        objs  = sep.extract(sub, 1.5, err=rms)
        radii = []
        for o in objs:
            if o["npix"] < 5:
                continue
            sigma = float(np.sqrt(o["a"] * o["b"]))
            radii.append(sigma * 2.0 * math.sqrt(2.0 * math.log(2.0)) * 0.5)
        return float(np.median(radii)) if radii else default
    except Exception:
        return default


# =============================================================================
# Low-level inference helpers
# =============================================================================

def _ort_pad(nchw: np.ndarray, mult: int = 32) -> tuple[np.ndarray, int, int]:
    """
    Pad a (1, C, H, W) float32 array so H and W are multiples of `mult`.
    Returns (padded, orig_H, orig_W).  Keeping a uniform tile shape prevents
    cuDNN from re-benchmarking convolution algorithms on every edge tile.
    """
    h, w = nchw.shape[2], nchw.shape[3]
    ph   = (-h) % mult
    pw   = (-w) % mult
    if ph or pw:
        nchw = np.pad(nchw, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    return nchw, h, w


def _infer_stellar(m: SharpenModels, chunk2d: np.ndarray) -> np.ndarray:
    chunk2d = np.asarray(chunk2d, np.float32)
    c, h0, w0 = _pad_mult16(chunk2d)          # Torch needs mult-16; ONNX gets mult-32 below

    if m.is_onnx:
        nchw       = np.tile(c[None, None, :, :], (1, 3, 1, 1))
        nchw, _, _ = _ort_pad(nchw)            # upgrade to mult-32; keep original h0/w0
        img_name, _, out_name = _ort_pick_io_names(m.stellar)
        raw = m.stellar.run([out_name], {img_name: nchw})[0]  # (1, 3, Hp, Wp)
        y   = raw[0, 0, :h0, :w0]
    else:
        t = torch.from_numpy(c).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(m.device)
        with torch.no_grad(), _autocast(m.device):
            y = m.stellar(t)[0, 0].detach().cpu().numpy()
        y = y[:h0, :w0]

    return y.astype(np.float32, copy=False)


def _infer_ns(m: SharpenModels, chunk2d: np.ndarray, psf01: float) -> np.ndarray:
    chunk2d = np.asarray(chunk2d, np.float32)
    c, h0, w0 = _pad_mult16(chunk2d)

    if m.is_onnx:
        nchw       = np.tile(c[None, None, :, :], (1, 3, 1, 1))
        nchw, _, _ = _ort_pad(nchw)            # keep original h0/w0
        img_name, psf_name, out_name = _ort_pick_io_names(m.ns_cond)
        if psf_name is None:
            raise RuntimeError(
                "Non-stellar ONNX model unexpectedly has no PSF input.")
        psf  = np.array([[float(psf01)]], dtype=np.float32)
        raw  = m.ns_cond.run([out_name], {img_name: nchw, psf_name: psf})[0]
        y    = raw[0, 0, :h0, :w0]
    else:
        t_rgb = torch.from_numpy(c).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(m.device)
        psf_t = torch.tensor([[float(psf01)]], dtype=torch.float32, device=m.device)
        with torch.no_grad(), _autocast(m.device):
            y = m.ns_cond(t_rgb, psf_t)[0, 0].detach().cpu().numpy()
        y = y[:h0, :w0]

    return y.astype(np.float32, copy=False)


def _infer_denoise_2d(m: DenoiseModels, model: Any,
                      chunk2d: np.ndarray) -> np.ndarray:
    chunk2d = np.asarray(chunk2d, np.float32)
    c, h0, w0 = _pad_mult16(chunk2d)

    if m.is_onnx:
        nchw       = np.tile(c[None, None, :, :], (1, 3, 1, 1))
        nchw, _, _ = _ort_pad(nchw)            # keep original h0/w0
        img_name   = model.get_inputs()[0].name
        out_name   = model.get_outputs()[0].name
        raw = model.run([out_name], {img_name: nchw})[0]
        y   = raw[0, 0, :h0, :w0]
    else:
        t = torch.from_numpy(c).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(m.device)
        with torch.no_grad(), _autocast(m.device):
            y = model(t)[0, 0].detach().cpu().numpy()
        y = y[:h0, :w0]

    return y.astype(np.float32, copy=False)


def _infer_denoise_rgb(m: DenoiseModels, model: Any,
                       chunk_rgb: np.ndarray) -> np.ndarray:
    chunk_rgb = np.asarray(chunk_rgb, np.float32)
    h0, w0    = chunk_rgb.shape[:2]

    # Torch needs mult-16
    ph16, pw16 = (-h0) % 16, (-w0) % 16
    cp = (np.pad(chunk_rgb, ((0, ph16), (0, pw16), (0, 0)), mode="reflect")
          if (ph16 or pw16) else chunk_rgb)
    nchw = np.ascontiguousarray(np.transpose(cp, (2, 0, 1))[None, ...])  # (1,3,H,W)

    if m.is_onnx:
        nchw, _, _ = _ort_pad(nchw)             # upgrade to mult-32; keep original h0/w0
        img_name = model.get_inputs()[0].name
        out_name = model.get_outputs()[0].name
        raw = model.run([out_name], {img_name: nchw})[0]  # (1,3,Hp,Wp)
        y   = np.transpose(raw[0], (1, 2, 0))[:h0, :w0]
    else:
        t = torch.from_numpy(nchw).to(m.device)
        with torch.no_grad(), _autocast(m.device):
            y = model(t)[0].detach().cpu().numpy()
        y = np.transpose(y, (1, 2, 0))[:h0, :w0]

    return y.astype(np.float32, copy=False)


def _infer_superres(m: SuperresModel, patch3c: np.ndarray) -> np.ndarray:
    patch3c = np.asarray(patch3c, np.float32)
    nchw    = np.ascontiguousarray(
                  np.transpose(patch3c, (2, 0, 1))[None, ...])  # (1,3,H,W)

    if m.is_onnx:
        nchw, _h, _w = _ort_pad(nchw)
        img_name = m.model.get_inputs()[0].name
        out_name = m.model.get_outputs()[0].name
        out = m.model.run([out_name], {img_name: nchw})[0]   # (1,3,H*s,W*s)
        # Super-res output is always larger — no crop needed
    else:
        t = torch.from_numpy(nchw).to(m.device)
        with torch.no_grad(), _autocast(m.device):
            out = m.model(t)[0].detach().cpu().numpy()

    return out[0]   # first channel, shape (H*scale, W*scale)


def _infer_darkstar(m: DarkstarModels, tile_rgb: np.ndarray) -> np.ndarray:
    tile_rgb = np.asarray(tile_rgb, np.float32)
    h0, w0   = tile_rgb.shape[:2]

    # Pad H and W to multiples of 16 first (padder_size = 2^4 = 16)
    ph = (-h0) % 16
    pw = (-w0) % 16
    if ph or pw:
        tile_rgb = np.pad(tile_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")

    nchw = np.ascontiguousarray(
               np.transpose(tile_rgb, (2, 0, 1))[None, ...])   # (1,3,Hp,Wp)

    if m.is_onnx:
        nchw, fh, fw = _ort_pad(nchw)          # upgrade to mult-32
        img_name = m.model.get_inputs()[0].name
        out_name = m.model.get_outputs()[0].name
        raw = m.model.run([out_name], {img_name: nchw})[0]
        y   = np.transpose(raw[0], (1, 2, 0))[:fh, :fw]
    else:
        t = torch.from_numpy(nchw).to(m.device)
        with torch.no_grad(), _autocast(m.device):
            y = m.model(t)[0].detach().cpu().numpy()
        y = np.transpose(y, (1, 2, 0))

    return y[:h0, :w0].astype(np.float32, copy=False)

# =============================================================================
# Processing pipelines
# =============================================================================

ProgressCB = Optional[Callable[[int, int, str], None]]  # (done, total, label)


def _check_cancel(cancel_cb: Optional[Callable[[], bool]]) -> None:
    if cancel_cb and cancel_cb():
        raise RuntimeError("Processing cancelled.")


# ---- Sharpen ----------------------------------------------------------------

def _sharpen_plane(
    m: SharpenModels, plane: np.ndarray, params: dict, psf_ref: np.ndarray,
    progress_cb: ProgressCB, label: str,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    chunk_size = int(params.get("chunk_size", 256))
    overlap    = int(params.get("overlap", 64))
    mode       = str(params.get("sharpening_mode", "Both"))
    s_amt      = float(params.get("stellar_amount", 0.5))
    ns_amt     = float(params.get("nonstellar_amount", 0.5))
    auto_psf   = bool(params.get("auto_psf", True))
    psf_r      = float(params.get("nonstellar_psf", 3.0))

    chunks = _split_chunks(plane, chunk_size, overlap)
    total  = len(chunks)

    if mode in ("Stellar Only", "Both"):
        out = []
        denom = total * (2 if mode == "Both" else 1)
        for k, (chunk, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            y = _infer_stellar(m, chunk)
            out.append((_blend(chunk, y, s_amt), i, j))
            if progress_cb:
                progress_cb(k, denom, f"{label} stellar")
        plane = _stitch_border_ignore(out, plane.shape)
        if mode == "Both":
            chunks = _split_chunks(plane, chunk_size, overlap)
            total  = len(chunks)

    if mode in ("Non-Stellar Only", "Both"):
        offset = total if mode == "Both" else 0
        denom  = total * (2 if mode == "Both" else 1)
        out    = []
        for k, (chunk, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            if auto_psf:
                h, w = chunk.shape
                ref  = psf_ref[i:i+h, j:j+w]
                r    = float(np.clip(_measure_psf(ref[:h, :w]), 1.0, 8.0))
            else:
                r = float(np.clip(psf_r, 1.0, 8.0))
            y = _infer_ns(m, chunk, _encode_psf(r))
            out.append((_blend(chunk, y, ns_amt), i, j))
            if progress_cb:
                progress_cb(offset + k, denom, f"{label} non-stellar")
        plane = _stitch_border_ignore(out, plane.shape)

    return plane


def process_sharpen(
    img_rgb01: np.ndarray, params: dict,
    progress_cb: ProgressCB = None,
    cancel_cb:   Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    use_gpu    = bool(params.get("use_gpu", True))
    separate   = bool(params.get("sharpen_channels_separately", False))
    t_stretch  = bool(params.get("temp_stretch", False))
    target_med = float(params.get("target_median", 0.25))

    m    = load_sharpen_models(use_gpu)
    img3 = np.clip(np.asarray(img_rgb01, np.float32), 0, 1)
    b    = _add_border_rgb(img3, 16)

    stretch_needed = t_stretch or float(np.median(b - b.min())) < 0.08
    if stretch_needed:
        b, omin, omeds = _stretch(b, target_med)
    else:
        omin, omeds = None, None

    psf_ref, _, _ = _extract_luminance(b)

    if separate:
        out = np.empty_like(b)
        for c, lbl in enumerate(("R", "G", "B")):
            out[..., c] = _sharpen_plane(m, b[..., c], params, psf_ref,
                                         progress_cb, lbl, cancel_cb)
        sharpened = out
    else:
        y, cb, cr = _extract_luminance(b)
        y2        = _sharpen_plane(m, y, params, psf_ref,
                                   progress_cb, "Luma", cancel_cb)
        sharpened = _merge_luminance(y2, cb, cr)

    if stretch_needed:
        sharpened = _unstretch(sharpened, omeds, omin, target_med)

    return np.clip(_remove_border(sharpened, 16), 0, 1).astype(np.float32, copy=False)


# ---- Denoise ----------------------------------------------------------------

def process_denoise(
    img_rgb01: np.ndarray, params: dict,
    progress_cb: ProgressCB = None,
    cancel_cb:   Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    use_gpu    = bool(params.get("use_gpu", True))
    lite       = bool(params.get("lite", False))
    walking    = bool(params.get("walking", False))
    mode       = str(params.get("denoise_mode", "full"))
    strength   = float(params.get("denoise_strength", 0.5))
    c_str      = float(params.get("denoise_color", strength))
    chunk_size = int(params.get("chunk_size", 256))
    overlap    = int(params.get("overlap", 64))
    t_stretch  = bool(params.get("temp_stretch", False))
    target_med = float(params.get("target_median", 0.25))

    m   = load_denoise_models(use_gpu, lite, walking)
    img = np.asarray(img_rgb01, np.float32)

    mono_input = (
        img.ndim == 2 or
        (img.ndim == 3 and img.shape[2] == 1) or
        (img.ndim == 3 and img.shape[2] == 3 and
         np.max(np.abs(img[..., 0] - img[..., 1])) <= 1e-7 and
         np.max(np.abs(img[..., 0] - img[..., 2])) <= 1e-7)
    )

    if mono_input:
        mono = img[..., 0] if img.ndim == 3 else img
        sn   = t_stretch or float(np.median(mono - mono.min())) < 0.05
        if sn:
            ms, omin, omeds = _stretch(mono, target_med)
        else:
            ms    = mono.astype(np.float32, copy=False)
            omin  = float(mono.min())
            omeds = [float(np.median(ms))]
        ms_b   = _add_border_2d(ms, 16)
        chunks = _split_chunks(ms_b, chunk_size, overlap)
        total  = len(chunks)
        out    = []
        for k, (chunk, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            out.append((_infer_denoise_2d(m, m.mono, chunk), i, j))
            if progress_cb:
                progress_cb(k, total, "Denoising")
        dm  = _stitch_border_ignore(out, ms_b.shape)
        res = np.clip(_blend(ms_b, dm, strength), 0, 1)
        if sn:
            res = _unstretch(res, omeds, omin, target_med)
        res = np.clip(_remove_border(res, 16), 0, 1).astype(np.float32)
        return np.stack([res, res, res], axis=-1)

    sn = t_stretch or float(np.median(img - img.min())) < 0.05
    if sn:
        img_s, omin, omeds = _stretch(img, target_med)
    else:
        img_s = img.astype(np.float32, copy=False)
        omin  = float(img.min())
        omeds = [float(np.median(img[..., c])) for c in range(3)]
    img_b = _add_border_rgb(img_s, 16)

    if mode == "separate":
        out_ch = []
        for c in range(3):
            chunks = _split_chunks(img_b[..., c], chunk_size, overlap)
            total  = len(chunks)
            out    = []
            for k, (chunk, i, j) in enumerate(chunks, 1):
                _check_cancel(cancel_cb)
                out.append((_infer_denoise_2d(m, m.mono, chunk), i, j))
                if progress_cb:
                    progress_cb(c * total + k, 3 * total, "Denoising")
            dc = _stitch_border_ignore(out, img_b[..., c].shape)
            out_ch.append(_blend(img_b[..., c], dc, strength))
        den = np.stack(out_ch, axis=-1)

    elif mode == "luminance":
        y, cb, cr = _extract_luminance(img_b)
        chunks    = _split_chunks(y, chunk_size, overlap)
        total     = len(chunks)
        out       = []
        for k, (chunk, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            out.append((_infer_denoise_2d(m, m.mono, chunk), i, j))
            if progress_cb:
                progress_cb(k, total, "Denoising luminance")
        dy  = _stitch_border_ignore(out, y.shape)
        den = _merge_luminance(_blend(y, dy, strength), cb, cr)

    else:  # full: luma via mono model, chroma via colour model
        y, cb, cr = _extract_luminance(img_b)

        chunks = _split_chunks(y, chunk_size, overlap)
        total_l = len(chunks)
        out_l   = []
        for k, (chunk, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            out_l.append((_infer_denoise_2d(m, m.mono, chunk), i, j))
            if progress_cb:
                progress_cb(k, total_l * 2, "Denoising luma")
        dy = _stitch_border_ignore(out_l, y.shape)
        y2 = _blend(y, dy, strength)

        chunks_rgb = _split_chunks(img_b, chunk_size, overlap)
        total_c    = len(chunks_rgb)
        out_c      = []
        for k, (chunk, i, j) in enumerate(chunks_rgb, 1):
            _check_cancel(cancel_cb)
            out_c.append((_infer_denoise_rgb(m, m.color, chunk), i, j))
            if progress_cb:
                progress_cb(total_l + k, total_l * 2, "Denoising colour")
        den_rgb          = _stitch_rgb_border_ignore(out_c, img_b.shape)
        _, cb_d, cr_d    = _extract_luminance(den_rgb)
        den = _merge_luminance(y2, _blend(cb, cb_d, c_str), _blend(cr, cr_d, c_str))

    den = np.clip(den, 0, 1)
    if sn:
        den = _unstretch(den, omeds, omin, target_med)
    return np.clip(_remove_border(den, 16), 0, 1).astype(np.float32, copy=False)


# ---- Super Resolution -------------------------------------------------------

def process_superres(
    img_rgb01: np.ndarray, params: dict,
    progress_cb: ProgressCB = None,
    cancel_cb:   Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    scale   = int(params.get("scale", 2))
    use_gpu = bool(params.get("use_gpu", True))
    border  = 16
    cs, ov  = 256, 64

    m = load_superres_model(scale, use_gpu)
    out_chans: list[np.ndarray] = []

    for c in range(3):
        _check_cancel(cancel_cb)
        chan = img_rgb01[..., c].astype(np.float32, copy=False)
        b    = _add_border_2d(chan, border)

        sn = float(np.median(b)) < 0.08
        if sn:
            bs, omin, omeds = _stretch(b)
        else:
            bs = b; omin = float(b.min()); omeds = [float(np.median(b))]

        # Bicubic pre-upscale via torch (no OpenCV dependency)
        h, w = bs.shape
        t_up = F.interpolate(
            torch.from_numpy(bs).unsqueeze(0).unsqueeze(0),
            size=(h * scale, w * scale), mode="bicubic", align_corners=False,
        ).squeeze().numpy().astype(np.float32)

        chunks = _split_chunks(t_up, cs, ov)
        total  = len(chunks)
        proc: list = []
        for k, (patch, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            ph, pw  = patch.shape
            patch3  = np.zeros((cs, cs, 3), np.float32)
            patch3[:ph, :pw, 0] = patch[:ph, :pw]
            patch3[:ph, :pw, 1] = patch[:ph, :pw]
            patch3[:ph, :pw, 2] = patch[:ph, :pw]
            out_p   = _infer_superres(m, patch3)
            proc.append((out_p[:ph, :pw], i, j))
            if progress_cb:
                progress_cb(c * total + k, 3 * total, f"Super-res ch {c+1}/3")

        stitched = _stitch_border_ignore(proc, t_up.shape)
        if sn:
            stitched = _unstretch(stitched, omeds, omin)
        out_chans.append(_remove_border(stitched, border * scale))

    return np.clip(np.stack(out_chans, axis=-1), 0, 1).astype(np.float32, copy=False)


# ---- Star Removal (DarkStar) ------------------------------------------------

def process_darkstar(
    img_rgb01: np.ndarray, params: dict,
    progress_cb: ProgressCB = None,
    cancel_cb:   Optional[Callable[[], bool]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Returns (starless, stars_only_or_None)."""
    use_gpu  = bool(params.get("use_gpu", True))
    chunk_sz = int(params.get("chunk_size", 512))
    overlap  = int(params.get("overlap", 64))
    mode     = str(params.get("darkstar_mode", "unscreen"))
    want_so  = bool(params.get("output_stars_only", False))
    border   = 5

    img = np.clip(np.asarray(img_rgb01, np.float32), 0, 1)
    same_rg  = np.allclose(img[..., 0], img[..., 1], rtol=0, atol=1e-6)
    same_rb  = np.allclose(img[..., 0], img[..., 2], rtol=0, atol=1e-6)
    is_color = not (same_rg and same_rb)

    m       = load_darkstar_models(use_gpu, is_color)

    sn = float(np.median(img - float(img.min()))) < 0.125
    if sn:
        img_s, omin, omeds = _stretch(img)
    else:
        img_s, omin, omeds = img, None, None

    b_img  = _add_border_rgb(img_s, border)
    chunks = _split_chunks(b_img, chunk_sz, overlap)
    total  = len(chunks)
    tiles: list = []

    for k, (tile, i, j) in enumerate(chunks, 1):
        _check_cancel(cancel_cb)
        tiles.append((_infer_darkstar(m, tile), i, j))
        if progress_cb:
            progress_cb(k, total, "Star removal")

    sl_b = _stitch_rgb_soft_blend(tiles, b_img.shape, chunk_sz, overlap, border)
    if sn:
        sl_b = _unstretch(sl_b, omeds, omin)
    starless = np.clip(_remove_border(sl_b, border), 0, 1).astype(np.float32, copy=False)

    so: Optional[np.ndarray] = None
    if want_so:
        if mode == "additive":
            so = np.clip(img - starless, 0, 1).astype(np.float32, copy=False)
        else:  # unscreen
            denom = np.maximum(1.0 - starless, 1e-6)
            so    = np.clip((img - starless) / denom, 0, 1).astype(np.float32, copy=False)

    return starless, so


# ---- Satellite Trail Removal ------------------------------------------------

def _infer_sat_classify(m: SatelliteModels, tile_rgb: np.ndarray) -> bool:
    """Run dual-classifier detection. Returns True if a satellite trail is present."""
    device = m.device
    # Resize to 256×256 for the classifiers (replicates torchvision ToTensor+Resize)
    t = torch.from_numpy(np.transpose(tile_rgb.astype(np.float32), (2, 0, 1))).unsqueeze(0)
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False).to(device)

    with torch.no_grad():
        o1 = float(m.detect1(t).item())
    if o1 <= 0.5:
        return False
    with torch.no_grad():
        o2 = float(m.detect2(t).item())
    return o2 > 0.25


def _infer_sat_remove(m: SatelliteModels, tile_rgb: np.ndarray) -> np.ndarray:
    tile_rgb = np.asarray(tile_rgb, np.float32)
    h0, w0   = tile_rgb.shape[:2]

    ph = (-h0) % 16
    pw = (-w0) % 16
    if ph or pw:
        tile_rgb = np.pad(tile_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")

    nchw = np.ascontiguousarray(
               np.transpose(tile_rgb, (2, 0, 1))[None, ...])   # (1,3,Hp,Wp)

    if m.is_onnx:
        nchw, fh, fw = _ort_pad(nchw)
        img_name = m.remover.get_inputs()[0].name
        out_name = m.remover.get_outputs()[0].name
        raw = m.remover.run([out_name], {img_name: nchw})[0]
        y   = np.transpose(raw[0], (1, 2, 0))[:fh, :fw]
    else:
        t = torch.from_numpy(nchw).to(m.device)
        with torch.no_grad(), _autocast(m.device):
            y = m.remover(t).squeeze(0).detach().cpu().numpy()
        y = np.transpose(y, (1, 2, 0))

    return np.clip(y[:h0, :w0], 0.0, 1.0).astype(np.float32, copy=False)

def _sat_clip_trail(processed: np.ndarray, original: np.ndarray,
                    sensitivity: float) -> np.ndarray:
    """Mask-based trail clipping from the original SASpro logic."""
    trail_only = original - processed
    mean_val   = float(np.mean(trail_only))
    clipped    = np.clip((trail_only - mean_val) * 10.0, 0.0, 1.0)
    mask       = np.where(clipped < sensitivity, 0.0, 1.0).astype(np.float32)
    return np.clip(original - mask, 0.0, 1.0)


def process_satellite(
    img_rgb01: np.ndarray, params: dict,
    progress_cb: ProgressCB = None,
    cancel_cb:   Optional[Callable[[], bool]] = None,
) -> tuple[np.ndarray, bool]:
    """
    Returns (cleaned_image, trail_was_detected).
    Luminance mode processes only the Y channel; full mode processes all three.
    """
    use_gpu     = bool(params.get("use_gpu", True))
    mode        = str(params.get("sat_mode", "full"))
    clip_trail  = bool(params.get("clip_trail", True))
    sensitivity = float(params.get("sensitivity", 0.1))
    chunk_size  = int(params.get("chunk_size", 256))
    overlap     = int(params.get("overlap", 64))
    t_stretch   = bool(params.get("temp_stretch", True))
    target_med  = float(params.get("target_median", 0.25))
    border      = 16

    m   = load_satellite_models(use_gpu)
    img = np.clip(np.asarray(img_rgb01, np.float32), 0, 1)

    def _should_stretch(x: np.ndarray) -> bool:
        return bool(np.median(x - float(x.min())) < 0.05)

    def _process_rgb(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        """Tile, detect, remove on a stretched HxWx3 array. Returns (result, detected)."""
        rgb_b  = _add_border_rgb(rgb, border)
        chunks = _split_chunks(rgb_b, chunk_size, overlap)
        total  = len(chunks)
        out_tiles: list = []
        detected_any = False

        for k, (tile, i, j) in enumerate(chunks, 1):
            _check_cancel(cancel_cb)
            tile_f = tile.astype(np.float32, copy=False)
            if _infer_sat_classify(m, tile_f):
                detected_any = True
                pred = _infer_sat_remove(m, tile_f)
                final = _sat_clip_trail(pred, tile_f, sensitivity) if clip_trail else pred
            else:
                final = tile_f
            out_tiles.append((final, i, j))
            if progress_cb:
                progress_cb(k, total, "Satellite removal")

        result = _stitch_rgb_border_ignore(out_tiles, rgb_b.shape, border=border)

        # Restore image edges unchanged (SASpro behaviour)
        if border > 0:
            result[:border,  :, :] = rgb_b[:border,  :, :]
            result[-border:, :, :] = rgb_b[-border:, :, :]
            result[:, :border,  :] = rgb_b[:, :border,  :]
            result[:, -border:, :] = rgb_b[:, -border:, :]

        result = np.clip(result, 0, 1)
        if not detected_any:
            return rgb_b.astype(np.float32, copy=False), False
        return result, True

    if mode == "luminance":
        y, cb, cr = _extract_luminance(img)
        sn = t_stretch or _should_stretch(y)
        if sn:
            y_s, omin, omeds = _stretch(y, target_med)
        else:
            y_s = y; omin = float(y.min()); omeds = [float(np.median(y))]

        y3 = np.stack([y_s, y_s, y_s], axis=-1)
        out3, detected = _process_rgb(y3)
        out_y = out3[..., 0]
        if sn:
            out_y = _unstretch(out_y, omeds, omin, target_med)
        out_y = np.clip(_remove_border(out_y, border), 0, 1)
        result = np.clip(_merge_luminance(out_y, cb, cr), 0, 1).astype(np.float32, copy=False)

    else:  # full
        sn = t_stretch or _should_stretch(img)
        if sn:
            img_s, omin, omeds = _stretch(img, target_med)
        else:
            img_s = img; omin = float(img.min()); omeds = [float(np.median(img[..., c])) for c in range(3)]

        out_rgb, detected = _process_rgb(img_s)
        if sn:
            out_rgb = _unstretch(out_rgb, omeds, omin, target_med)
        result = np.clip(_remove_border(out_rgb, border), 0, 1).astype(np.float32, copy=False)

    return result, detected

# =============================================================================
# Siril image I/O helpers
# =============================================================================

def siril_get_image(siril) -> tuple[np.ndarray, bool, bool]:
    """
    Fetch image from Siril and normalise to HxWx3 float32 [0,1].
    Returns (img_hwx3, was_mono, was_planar).
    Siril can return either (H,W), (H,W,C) or (C,H,W).
    """
    raw       = siril.get_image_pixeldata()
    orig_dtype = raw.dtype
    is_uint16  = (orig_dtype == np.uint16)

    if is_uint16:
        raw = raw.astype(np.float32) / 65535.0
    else:
        raw = raw.astype(np.float32, copy=False)

    was_planar = False
    was_mono   = False

    if raw.ndim == 2:
        was_mono = True
        img      = np.stack([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[0] in (1, 3) and raw.shape[0] < raw.shape[1]:
        # Channels-first (C,H,W)
        was_planar = True
        was_mono   = raw.shape[0] == 1
        img        = np.transpose(raw, (1, 2, 0))
        if was_mono:
            img = np.concatenate([img, img, img], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 1:
        was_mono = True
        img      = np.concatenate([raw, raw, raw], axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        img = raw
    else:
        raise ValueError(f"Unsupported pixel data shape: {raw.shape}")

    return np.clip(img, 0, 1), was_mono, was_planar, is_uint16


def siril_set_image(siril, result: np.ndarray,
                    was_mono: bool, was_planar: bool, is_uint16: bool) -> None:
    """
    Write processed HxWx3 float32 array back to Siril in the original format.
    """
    out = result.astype(np.float32, copy=False)

    if was_mono:
        out = out[..., 0]          # back to 2-D

    if was_planar and out.ndim == 3:
        out = np.transpose(out, (2, 0, 1))

    if is_uint16:
        out = (np.clip(out, 0, 1) * 65535.0).astype(np.uint16)

    with siril.image_lock():
        siril.set_image_pixeldata(out)


# =============================================================================
# Worker thread (keeps the GUI responsive during processing)
# =============================================================================

class ProcessWorker(QThread):
    progress_signal = pyqtSignal(int, int, str)   # done, total, label
    log_signal      = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)        # success, message

    def __init__(self, mode: str, params: dict, siril):
        super().__init__()
        self.mode   = mode
        self.params = params
        self.siril  = siril
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self) -> bool:
        return self._cancel

    def _progress(self, done: int, total: int, label: str) -> None:
        self.progress_signal.emit(done, total, label)

    def run(self):
        try:
            img, was_mono, was_planar, is_uint16 = siril_get_image(self.siril)

            self.log_signal.emit(
                f"Image: {img.shape[1]}×{img.shape[0]} px  "
                f"{'mono' if was_mono else 'colour'}  "
                f"{'uint16' if is_uint16 else 'float32'}"
            )

            self.siril.undo_save_state(f"Cosmic Clarity – {self.mode}")

            if self.mode == "sharpen":
                result = process_sharpen(img, self.params,
                                            progress_cb=self._progress,
                                            cancel_cb=self._cancelled)
                siril_set_image(self.siril, result, was_mono, was_planar, is_uint16)

            elif self.mode in ("denoise", "denoise_lite", "denoise_1w"):
                p = dict(self.params)
                p["lite"]    = (self.mode == "denoise_lite")
                p["walking"] = (self.mode == "denoise_1w")
                result = process_denoise(img, p,
                                            progress_cb=self._progress,
                                            cancel_cb=self._cancelled)
                siril_set_image(self.siril, result, was_mono, was_planar, is_uint16)

            elif self.mode == "superres":
                result = process_superres(img, self.params,
                                            progress_cb=self._progress,
                                            cancel_cb=self._cancelled)
                # result is (H*scale, W*scale, 3) — sirilpy expects (C, H, W)
                out = np.transpose(result.astype(np.float32, copy=False), (2, 0, 1))
                with self.siril.image_lock():
                    self.siril.set_image_pixeldata(out)

            elif self.mode == "darkstar":
                starless, so = process_darkstar(img, self.params,
                                                progress_cb=self._progress,
                                                cancel_cb=self._cancelled)
                siril_set_image(self.siril, starless, was_mono, was_planar, is_uint16)

                if so is not None:
                    image = self.siril.get_image(with_pixels=False) # Get metadata
                    # Update filter metadata and save the image
                    lines = image.header.split('\n')
                    lines = [line for line in lines if not line.strip().startswith('END')]
                    while lines and not lines[-1].strip():
                        lines.pop()
                    header_dict = fits.Header.fromstring('\n'.join(lines), sep='\n')
                    header_dict["FILTER"] = "starless"
                    hdu = fits.PrimaryHDU(header=header_dict)
                    hdu.header.remove('BG-PTS', ignore_missing=True) # Remove broken GraXpert header card if present
                    hdu.verify('silentfix')
                    header_str = hdu.header.tostring(sep='\n')
                    orig_filename = self.siril.get_image_filename()
                    savename = "starmask_" + Path(orig_filename).name
                    if was_mono:
                        so = so[..., 0]          # back to 2-D

                    if was_planar and so.ndim == 3:
                        so = np.transpose(so, (2, 0, 1))

                    if is_uint16:
                        so = (np.clip(so, 0, 1) * 65535.0).astype(np.uint16)
                    self.siril.save_image_file(so, header_str, filename=savename)
                    self.log_signal.emit(
                        f"Starmask layer computed and saved as {savename}."
                    )

            elif self.mode == "satellite":
                result, detected = process_satellite(img, self.params,
                                                     progress_cb=self._progress,
                                                     cancel_cb=self._cancelled)
                siril_set_image(self.siril, result, was_mono, was_planar, is_uint16)
                if not detected:
                    self.log_signal.emit("No satellite trails detected in this image.")

            if self._cancel:
                self.finished_signal.emit(False, "Cancelled.")
            else:
                self.siril.log(f"Cosmic Clarity [{self.mode}] complete.")
                self.finished_signal.emit(True, "Done.")

        except RuntimeError as exc:
            if self._cancel or str(exc) == "Processing cancelled.":
                self.finished_signal.emit(False, "Cancelled.")
            else:
                self.finished_signal.emit(False, str(exc))
        except Exception as exc:
            tb = traceback.format_exc()
            self.finished_signal.emit(False, f"{exc}\n\n{tb}")


# =============================================================================
# Model download worker
# =============================================================================

class DownloadWorker(QThread):
    log_signal      = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, zip_dst: Path):
        super().__init__()
        self.zip_dst = zip_dst
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            download_models(                          # ← was download_gdrive(GDRIVE_FILE_ID, ...)
                self.zip_dst,
                progress_cb=lambda msg: self.log_signal.emit(msg),
                cancel_cb=lambda: self._cancel,
            )
            if self._cancel:
                self.finished_signal.emit(False, "Download cancelled.")
                return
            install_models_zip(
                self.zip_dst,
                progress_cb=lambda msg: self.log_signal.emit(msg),
            )
            try:
                self.zip_dst.unlink(missing_ok=True)
            except Exception:
                pass
            self.finished_signal.emit(True, "Models downloaded and installed.")
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))

class ObsoleteWorker(QThread):
    log_signal      = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def run(self):
        try:
            n = remove_obsolete_models(
                progress_cb=lambda m: self.log_signal.emit(m))
            self.finished_signal.emit(True,
                f"Removed {n} obsolete file(s)."
                if n else "No obsolete model files found.")
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))

class PruneWorker(QThread):
    log_signal      = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)   # ok, summary

    def __init__(self, target: str):
        """target: 'onnx' or 'torch'"""
        super().__init__()
        self.target = target

    def run(self):
        try:
            if self.target == "onnx":
                n = remove_onnx_models(
                    progress_cb=lambda m: self.log_signal.emit(m))
                self.finished_signal.emit(True, f"Removed {n} ONNX file(s).")
            else:
                n = remove_torch_models(
                    progress_cb=lambda m: self.log_signal.emit(m))
                self.finished_signal.emit(True, f"Removed {n} Torch file(s).")
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))

# =============================================================================
# GUI
# =============================================================================

def _label(text: str, bold: bool = False, pt: int = 0) -> QLabel:
    lbl  = QLabel(text)
    font = lbl.font()
    if bold: font.setBold(True)
    if pt:   font.setPointSize(pt)
    lbl.setFont(font)
    return lbl


def _slider_row(parent_layout, label: str, lo: int, hi: int,
                default: int, scale: float = 100.0
                ) -> tuple[QSlider, QLabel]:
    row   = QHBoxLayout()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(lo)
    slider.setMaximum(hi)
    slider.setValue(default)
    val_lbl = QLabel(f"{default / scale:.2f}")
    val_lbl.setMinimumWidth(40)
    slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v / scale:.2f}"))
    row.addWidget(QLabel(label))
    row.addWidget(slider)
    row.addWidget(val_lbl)
    parent_layout.addLayout(row)
    return slider, val_lbl

class HelpDialog(QDialog):
    """Reference dialog covering all controls and concepts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cosmic Clarity AI4 for Siril — Help")
        self.setModal(True)
        self.resize(620, 580)

        layout = QVBoxLayout(self)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setReadOnly(True)
        browser.setHtml("""
<style>
  body  { font-family: sans-serif; font-size: 10pt; margin: 4px; }
  h2    { color: #ffffff; background: #1a237e; padding: 3px 6px;
          margin-top: 14px; margin-bottom: 2px; }
  h3    { color: #ffffff; background: #37474f; padding: 2px 6px;
          margin-top: 10px; margin-bottom: 2px; }
  p, li { margin-top: 2px; margin-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 4px; }
  th    { background: #37474f; color: #ffffff; padding: 3px 6px; text-align: left; }
  td    { padding: 3px 6px; border-bottom: 1px solid #e0e0e0; }
  code  { font-family: monospace; font-size: 9.5pt; font-weight: bold; }
  .note { color: #555; font-style: italic; }
  .warn { color: #b71c1c; }
</style>

<h2>Overview</h2>
<p>Cosmic Clarity for Siril runs the <b>Cosmic Clarity AI models</b> (by Franklin Marek /
SetiAstro) directly inside Siril, without needing a separate SetiAstroSuite Pro
installation. Select a tab for the operation you want to perform, adjust parameters,
and click <b>Run</b>. The result replaces the currently loaded image.</p>
<p>Models must be downloaded once via the <b>⬇ Models</b> tab before any processing
can be performed.</p>

<h2>Sharpen tab</h2>
<table>
  <tr><th>Control</th><th>Description</th></tr>
  <tr><td><b>Mode</b></td>
      <td><i>Both</i> runs stellar sharpening first, then non-stellar on the result.
          Use <i>Stellar Only</i> or <i>Non-Stellar Only</i> to target just one
          component.</td></tr>
  <tr><td><b>Stellar Amount</b></td>
      <td>Blend factor between the original and the stellar-sharpened output.
          0 = no change, 1 = full AI output.</td></tr>
  <tr><td><b>Non-Stellar Amount</b></td>
      <td>Same blend factor for the non-stellar (detail / nebulosity) pass.</td></tr>
  <tr><td><b>Auto-detect PSF</b></td>
      <td>Measures the local star radius in each tile using source extraction (SEP)
          and passes it to the non-stellar model as a conditioning value. Recommended.
          Requires the <code>sep</code> package; falls back to the manual value if
          SEP is unavailable.</td></tr>
  <tr><td><b>Non-Stellar PSF radius</b></td>
      <td>Manual PSF radius (1–8 pixels) used when auto-detect is off.</td></tr>
  <tr><td><b>Sharpen channels separately</b></td>
      <td>For colour images: sharpen R, G and B independently rather than
          operating on luminance only. Slower but can recover more colour detail.</td></tr>
  <tr><td><b>Force pre-stretch</b></td>
      <td>Apply a midtone stretch before sharpening and invert it afterwards.
          Needed for linear (unstretched) or very dark images where the model
          would otherwise see near-zero input.</td></tr>
</table>

<h2>Denoise tab</h2>
<table>
  <tr><th>Control</th><th>Description</th></tr>
  <tr><td><b>Mode</b></td>
      <td><i>full</i> — denoises all three channels simultaneously using the
          colour model.<br>
          <i>luminance</i> — denoises only the Y (luma) channel; preserves
          colour noise.<br>
          <i>separate</i> — denoises each RGB channel independently with the
          mono model.</td></tr>
  <tr><td><b>Luma Strength</b></td>
      <td>Blend factor applied to the denoised luminance (or mono) output.</td></tr>
  <tr><td><b>Colour Strength</b></td>
      <td>Blend factor applied to the denoised colour channels (full mode only).</td></tr>
  <tr><td><b>Use Lite model</b></td>
      <td>Loads the smaller NAFNet variant (half the channel width). Faster and
          uses less VRAM; quality is slightly reduced for very noisy images.</td></tr>
  <tr><td><b>Force pre-stretch</b></td>
      <td>As per Sharpen — recommended for linear data.</td></tr>
</table>

<h2>Super-Res tab</h2>
<p>Upscales the image by the chosen factor (<b>2×</b>, <b>3×</b> or <b>4×</b>)
using a two-stage process: bicubic upsampling followed by AI detail enhancement.
The output image is larger than the input; Siril will reflect the new dimensions.
Super-resolution does not use tiling — the entire image is processed at once, so
large images require more VRAM.</p>

<h2>Star Removal tab</h2>
<table>
  <tr><th>Control</th><th>Description</th></tr>
  <tr><td><b>Compute stars-only layer</b></td>
      <td>In addition to the starless result, calculates and saves a stars-only
          image (the difference between the original and the starless) as
          <code>starmask_&lt;filename&gt;</code> in the same directory.</td></tr>
  <tr><td><b>Compositing mode</b></td>
      <td><i>unscreen</i> — mathematically correct inverse of the screen
          blend mode; use this to re-add stars to the starless image later.<br>
          <i>additive</i> — simple pixel difference (original − starless).</td></tr>
</table>
<p class="note">The AI4 DarkStar model uses a NAFNet architecture. Separate mono
and colour checkpoints are provided; the correct one is selected automatically.</p>

<h2>Satellite Removal tab</h2>
<table>
  <tr><th>Control</th><th>Description</th></tr>
  <tr><td><b>Mode</b></td>
      <td><i>full</i> — detect and remove trails across all three channels.<br>
          <i>luminance</i> — process only the Y channel; faster and avoids
          colour fringing on faint trails.</td></tr>
  <tr><td><b>Clip trail</b></td>
      <td>After removal, clip any remaining trail brightness back towards
          the local background. Recommended for most images.</td></tr>
  <tr><td><b>Sensitivity</b></td>
      <td>Controls how aggressively the clip is applied. Lower values clip
          more of the residual trail; higher values only clip very bright
          remnants.</td></tr>
  <tr><td><b>Force pre-stretch</b></td>
      <td>Required for linear data — the trail detectors expect a
          stretched image.</td></tr>
</table>
<p class="note">Satellite removal requires <code>torchvision</code>, which is
normally installed alongside PyTorch. A warning is shown in the tab if it is
missing.</p>

<h2>⬇ Models tab</h2>
<p>All models are downloaded as a single zip archive (~2.4 GB) from HuggingFace
and extracted to your Siril user-data directory.</p>

<h3>Models directory</h3>
<p>By default models are stored in the <code>cosmic_clarity</code> subfolder of
your Siril user-data directory, shown in the location line at the top of this
tab. If you already have the Cosmic Clarity models installed elsewhere — for
example, inside a <b>SetiAstroSuite Pro</b> installation — you can point the
script at that directory using the <b>Browse…</b> button instead of downloading
a second copy. Click <b>Reset</b> to revert to the default location at any
time. The chosen path is saved in your settings.</p>

<h3>Downloading</h3>
<p>Click <b>Download Models (~2.4GB MB)</b> to fetch and install the full model
archive into the currently active directory. The download is from HuggingFace
and can be cancelled and restarted without losing progress.</p>
<p>Only a full model pack is provided. If a new model is released you may be
able to save download time by checking the <a href="https://drive.google.com/drive/folders/1-fktZb3I9l-mQimJX2fZAmJCBj_t0yAF">SetiAstro models Google drive</a>
to see if the new model is available for download separately. If so, you will
need to unpack it yourself into the model directory shown in 'Location' in the
Models tab.</p>

<h3>Removing models</h3>
<p><b>Remove all ONNX models</b> / <b>Remove all Torch models</b> — deletes
<code>.onnx</code> or <code>.pth</code>/<code>.pt</code> files from the models
directory. For safety, a file is only removed if its counterpart in the other
format is also present — so you will never be left with no working model for
any operation. You can re-download the full archive at any time.</p>

<h3>Model status indicators</h3>
<p>The per-tab model status grids show, for each model file:</p>
<ul>
  <li><b>PTH</b> — whether the Torch checkpoint is present (green ●) or
      missing (red ○).</li>
  <li><b>ONNX</b> — whether the ONNX file is present, and whether ONNX
      Runtime is installed. Shows <i>no ORT</i> if the runtime is absent.</li>
  <li><b>Backend</b> — which backend will actually be used when you click
      Run, given the current GPU setting, backend preference, and available
      files.</li>
</ul>

<h2>⚙ Settings tab</h2>
<table>
  <tr><th>Control</th><th>Description</th></tr>
  <tr><td><b>Tile size</b></td>
      <td>Images are processed in overlapping square tiles. Larger tiles give
          smoother results but require more VRAM. Must be a multiple of 32;
          128–1024 px.</td></tr>
  <tr><td><b>Overlap</b></td>
      <td>How many pixels of each tile edge are discarded before stitching.
          Larger overlaps reduce seam artefacts at the cost of more computation.
          Must be a multiple of 32 and less than half the tile size.</td></tr>
  <tr><td><b>Use GPU</b></td>
      <td>Enable GPU acceleration. The detected device is shown below the
          checkbox. Backends are tried in order: CUDA → MPS → XPU → DirectML
          → CPU. Disabling this forces CPU execution for all operations.</td></tr>
</table>

<h2>Backend selection</h2>
<p>Two inference backends are supported:</p>
<ul>
  <li><b>Torch</b> — PyTorch is used for CUDA (NVIDIA), MPS (Apple Silicon) and
      XPU (Intel Arc) devices, where it makes full use of the accelerator.</li>
  <li><b>ONNX Runtime (ORT)</b> — preferred on CPU and DirectML (Windows
      integrated / discrete GPU via DirectX 12). Requires the
      <code>onnxruntime</code> package (or <code>onnxruntime-directml</code>
      on Windows) to be installed in the Siril Python environment. When ORT is
      available and the ONNX file is present, it will be chosen automatically
      over Torch for those backends. Note that these onnxruntime models run
      poorly on CUDA systems, so Torch is strongly recommended on those.</li>
</ul>

<h2>CLI usage</h2>
<p>The script can be run headlessly from Siril's command line:</p>
<p><code>pyscript CosmicClarity_Native.py --mode sharpen --stellar-amount 0.6</code></p>
<p>Run with <code>--help</code> for the full list of options.</p>

<h2>Reporting bugs</h2>
<p>This script is not written by SetiAstro. Please report bugs at:<br>
<a href="https://gitlab.com/free-astro/siril-scripts">
https://gitlab.com/free-astro/siril-scripts</a></p>
""")
        layout.addWidget(browser)

        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn.accepted.connect(self.accept)
        layout.addWidget(btn)

class AboutDialog(QDialog):
    """Popup dialog showing background information about the script."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cosmic Clarity AI4 for Siril — About")
        self.setModal(True)
        self.resize(520, 400)

        layout = QVBoxLayout(self)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(True)
        browser.setReadOnly(True)
        browser.setHtml("""
<style>
  body  { font-family: sans-serif; font-size: 10pt; margin: 4px; }
  h2    { color: #ffffff; background: #1a237e; padding: 3px 6px;
          margin-top: 14px; margin-bottom: 2px; }
  h3    { color: #ffffff; background: #37474f; padding: 2px 6px;
          margin-top: 10px; margin-bottom: 2px; }
  p, li { margin-top: 2px; margin-bottom: 4px; }
  code  { font-family: monospace; font-size: 9.5pt; font-weight: bold; }
  .note { color: #555; font-style: italic; }
</style>

<h2>Cosmic Clarity</h2>
<p>Cosmic Clarity is a suite of AI image-processing tools written by
<b>Franklin Marek (SetiAstro)</b>, covering sharpening, denoising,
star removal, super-resolution and satellite trail removal.</p>
<p>Originally released as standalone programs, the Cosmic Clarity models
have since been incorporated as a core feature of
<b>SetiAstroSuite Pro</b>.</p>
<p>SetiAstro website:
<a href="https://www.setiastro.com">https://www.setiastro.com</a></p>
<p>SetiAstroSuite Pro and Cosmic Clarity models are © Franklin Marek.</p>

<h2>This script</h2>
<p>This Siril adaptation provides direct inferencing of the Cosmic Clarity
models without requiring a SetiAstroSuite Pro installation or its separate
Python environment. It is a lightweight alternative that may save over a
gigabyte of storage for users who do not need the full suite.</p>
<p>Both <code>torch</code> and <code>onnxruntime</code> backends are
supported. ONNX Runtime is preferred on CPU and DirectML; PyTorch is used
for CUDA, MPS and XPU devices. The active backend for each model is shown
in the per-tab status grids.</p>

<h3>Siril adaptation</h3>
<p>© Adrian Knagg-Baugh 2026. This script is not written by SetiAstro —
please report bugs at:<br>
<a href="https://gitlab.com/free-astro/siril-scripts">
https://gitlab.com/free-astro/siril-scripts</a></p>

<h3>Licence</h3>
<p>SPDX-License-Identifier: <code>GPL-3.0-or-later</code></p>
""")
        layout.addWidget(browser)

        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn.accepted.connect(self.accept)
        layout.addWidget(btn)

# =============================================================================
# Model availability metadata (used by GUI status panels)
# =============================================================================

# (display_name, pth_key, onnx_key_or_None)
_MODE_MODEL_INFO: dict[str, list[tuple[str, str, Optional[str]]]] = {
    "sharpen": [
        ("Stellar",      "sharpen_stellar",    "sharpen_stellar_onnx"),
        ("Non-stellar",  "sharpen_nonstellar", "sharpen_nonstellar_onnx"),
    ],
    "denoise": [
        ("Mono  (full)",    "denoise_mono",       "denoise_mono_onnx"),
        ("Color (full)",    "denoise_color",      "denoise_color_onnx"),
        ("Mono  (lite)",    "denoise_mono_lite",  "denoise_mono_lite_onnx"),
        ("Color (lite)",    "denoise_color_lite", "denoise_color_lite_onnx"),
        ("Mono  (walk.)",   "denoise_mono_1w",    "denoise_mono_1w_onnx"),
        ("Color (walk.)",   "denoise_color_1w",   "denoise_color_1w_onnx"),
    ],
    "superres": [
        ("2× scale", "superres_2x", "superres_2x_onnx"),
        ("3× scale", "superres_3x", "superres_3x_onnx"),
        ("4× scale", "superres_4x", "superres_4x_onnx"),
    ],
    "darkstar": [
        ("Mono",  "darkstar_mono",  "darkstar_mono_onnx"),
        ("Color", "darkstar_color", "darkstar_color_onnx"),
    ],
    "satellite": [
        ("Detector 1 (ResNet)",     "sat_detect1", None),
        ("Detector 2 (MobileNet)",  "sat_detect2", None),
        ("Remover",                 "sat_remove",  None),
    ],
}

# Column header labels shown in every status grid
_COL_HEADERS = ("Model", "PTH", "ONNX", "Backend")


def _row_state(pth_key: str, onnx_key: Optional[str],
               use_gpu: bool) -> tuple[bool, bool, bool, str]:
    pth_ok       = model_path(pth_key).exists() if pth_key in MODEL_FILES else False
    onnx_file_ok = (model_path(onnx_key).exists()
                    if onnx_key and onnx_key in MODEL_FILES else False)
    ort_installed = _get_ort() is not None

    if not pth_ok and not onnx_file_ok:
        preferred = "—"
    elif _BACKEND_PREF == "onnx":
        if ort_installed and onnx_file_ok:
            preferred = "ONNX"
        elif pth_ok:
            preferred = "Torch"       # ONNX preferred but unavailable — fall back
        else:
            preferred = "—"
    elif _BACKEND_PREF == "torch":
        if pth_ok:
            preferred = "Torch"
        elif ort_installed and onnx_file_ok:
            preferred = "ONNX"        # Torch preferred but unavailable — fall back
        else:
            preferred = "—"
    else:  # "auto"
        if ort_installed and onnx_file_ok:
            try:
                dev = get_device(use_gpu)
                prefer_onnx = dev.type not in ("cuda", "mps", "xpu")
            except Exception:
                prefer_onnx = True
            preferred = "ONNX" if prefer_onnx else "Torch"
        else:
            preferred = "Torch" if pth_ok else "—"

    return pth_ok, onnx_file_ok, ort_installed, preferred

def _apply_row_state(pth_lbl: "QLabel", onnx_lbl: "QLabel",
                     pref_lbl: "QLabel",
                     pth_key: str, onnx_key: Optional[str],
                     use_gpu: bool) -> None:
    """Update the three display labels for one model row."""
    pth_ok, onnx_ok, ort_ok, preferred = _row_state(pth_key, onnx_key, use_gpu)

    # PTH indicator
    if pth_ok:
        pth_lbl.setText("● PTH")
        pth_lbl.setStyleSheet("color:#2e7d32; font-weight:bold;")
    else:
        pth_lbl.setText("○ PTH")
        pth_lbl.setStyleSheet("color:#b71c1c;")

    # ONNX indicator
    if onnx_key is None:
        onnx_lbl.setText("—")
        onnx_lbl.setStyleSheet("color:#9e9e9e;")
    elif not ort_ok:
        onnx_lbl.setText("no ORT")
        onnx_lbl.setStyleSheet("color:#9e9e9e; font-style:italic;")
    elif onnx_ok:
        onnx_lbl.setText("● ONNX")
        onnx_lbl.setStyleSheet("color:#2e7d32; font-weight:bold;")
    else:
        onnx_lbl.setText("○ ONNX")
        onnx_lbl.setStyleSheet("color:#e65100;")

    # Preferred backend
    if preferred == "ONNX":
        pref_lbl.setText("→ ONNX")
        pref_lbl.setStyleSheet("color:#1565c0; font-weight:bold;")
    elif preferred == "Torch":
        pref_lbl.setText("→ Torch")
        pref_lbl.setStyleSheet("color:#4527a0; font-weight:bold;")
    else:
        pref_lbl.setText("⚠ missing")
        pref_lbl.setStyleSheet("color:#b71c1c; font-weight:bold;")

class CosmicClarityGUI(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril   = siril
        self.worker  = None
        self.dl_worker = None
        self.prune_worker = None
        self.setWindowTitle(f"Cosmic Clarity AI4 for Siril  v{VERSION}")
        self._mode_model_labels: dict = {}
        self._build_ui()
        self._load_settings()
        self._settings_saved = False
        self._settings_cache: dict = {}
        atexit.register(self._save_settings)
        self._refresh_model_status()

    def _sync_settings(self) -> None:
        """Copy current widget values into _settings_cache (no-op if widgets gone)."""
        try:
            self._settings_cache = {
                "use_gpu":              self.gpu_chk.isChecked(),
                "tile_size":            self.tile_size_spin.value(),
                "overlap":              self.overlap_spin.value(),
                "backend_pref_idx":     self.backend_combo.currentIndex(),
                "sharpen_mode":         self.sharpen_mode_combo.currentText(),
                "stellar_amount":       self.stellar_slider.value(),
                "ns_amount":            self.ns_slider.value(),
                "auto_psf":             self.auto_psf_chk.isChecked(),
                "psf_radius":           self.psf_spin.value(),
                "sep_channels":         self.sep_channels_chk.isChecked(),
                "sharpen_stretch":      self.sharpen_stretch_chk.isChecked(),
                "denoise_mode":         self.denoise_mode_combo.currentText(),
                "denoise_luma":         self.denoise_luma_slider.value(),
                "denoise_color":        self.denoise_color_slider.value(),
                "denoise_variant":       self.denoise_variant_combo.currentText(),
                "denoise_stretch":      self.denoise_stretch_chk.isChecked(),
                "superres_scale_idx":   self.scale_combo.currentIndex(),
                "darkstar_stars_only":  self.ds_stars_only_chk.isChecked(),
                "darkstar_mode":        self.ds_mode_combo.currentText(),
                "sat_mode":             self.sat_mode_combo.currentText(),
                "sat_clip":             self.sat_clip_chk.isChecked(),
                "sat_sensitivity":      self.sat_sens_slider.value(),
                "sat_stretch":          self.sat_stretch_chk.isChecked(),
                "models_dir_override":  str(models_dir_override() or ""),
            }
        except RuntimeError:
            pass  # widgets already destroyed — keep whatever is in cache

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(12)
        vbox.setContentsMargins(16, 16, 16, 16)

        # Title
        title = _label("Cosmic Clarity AI4 for Siril", bold=True, pt=14)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)
        vbox.addWidget(_label(f"v{VERSION}  –  AI sharpening · denoising · super-res · star removal · satellite removal", pt=9))
        vbox.addWidget(_label("AI models © Franklin Marek from SetiAstro (https://www.setiastro.com)", pt=8))

        btn_row = QHBoxLayout()
        self.help_btn  = QPushButton("Help")
        self.about_btn = QPushButton("About")
        self.help_btn.clicked.connect(self.show_help)
        self.about_btn.clicked.connect(self.show_about)
        btn_row.addWidget(self.help_btn)
        btn_row.addWidget(self.about_btn)
        vbox.addLayout(btn_row)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        vbox.addWidget(sep)

        # Tabs for each mode
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_sharpen(),   "Sharpen")
        self.tabs.addTab(self._tab_denoise(),   "Denoise")
        self.tabs.addTab(self._tab_superres(),  "Super-Res")
        self.tabs.addTab(self._tab_darkstar(),  "Star Removal")
        self.tabs.addTab(self._tab_satellite(), "Satellite")
        self.tabs.addTab(self._tab_models(),    "⬇ Models")
        self.tabs.addTab(self._tab_settings(),  "⚙ Settings")
        vbox.addWidget(self.tabs)

        # Progress area
        prog_box = QGroupBox("Progress")
        prog_vbox = QVBoxLayout(prog_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_vbox.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready.")
        self.progress_label.setStyleSheet("QLabel { color: #555; font-style: italic; }")
        prog_vbox.addWidget(self.progress_label)
        vbox.addWidget(prog_box)

        # Run / Cancel button
        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(40)
        f = self.run_btn.font(); f.setBold(True); f.setPointSize(11)
        self.run_btn.setFont(f)
        self.run_btn.clicked.connect(self._on_run_cancel)
        vbox.addWidget(self.run_btn)

        self._connect_settings_signals()

        self.adjustSize()
        self.setMinimumWidth(600)

    def _connect_settings_signals(self) -> None:
        """Wire every settings-bearing widget to _sync_settings so the cache
        stays current even if the window is closed without a clean Qt shutdown."""
        for widget in (
            self.gpu_chk, self.auto_psf_chk, self.sep_channels_chk,
            self.sharpen_stretch_chk,
            self.denoise_stretch_chk, self.ds_stars_only_chk,
            self.sat_clip_chk, self.sat_stretch_chk,
        ):
            widget.stateChanged.connect(lambda _: self._sync_settings())

        for widget in (
            self.sharpen_mode_combo, self.denoise_mode_combo,
            self.denoise_variant_combo, self.scale_combo,
            self.ds_mode_combo, self.sat_mode_combo, self.backend_combo,
        ):
            widget.currentIndexChanged.connect(lambda _: self._sync_settings())

        for widget in (
            self.stellar_slider, self.ns_slider,
            self.denoise_luma_slider, self.denoise_color_slider,
            self.sat_sens_slider,
        ):
            widget.valueChanged.connect(lambda _: self._sync_settings())

        for widget in (self.tile_size_spin, self.overlap_spin):
            widget.valueChanged.connect(lambda _: self._sync_settings())

        self.psf_spin.valueChanged.connect(lambda _: self._sync_settings())

    # ------------------------------------------------------------------ tabs

    def _tab_sharpen(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(self._build_mode_model_box("sharpen"))   # ← ADD
        vbox.addWidget(_label("Sharpening Mode"))
        self.sharpen_mode_combo = QComboBox()
        self.sharpen_mode_combo.addItems(["Both", "Stellar Only", "Non-Stellar Only"])
        vbox.addWidget(self.sharpen_mode_combo)

        self.stellar_slider, _  = _slider_row(vbox, "Stellar Amount:", 0, 100, 50)
        self.ns_slider, _       = _slider_row(vbox, "Non-Stellar Amount:", 0, 100, 50)

        self.auto_psf_chk = QCheckBox("Auto-detect PSF (recommended)")
        self.auto_psf_chk.setChecked(True)
        self.auto_psf_chk.stateChanged.connect(self._on_auto_psf_toggled)
        vbox.addWidget(self.auto_psf_chk)

        psf_row = QHBoxLayout()
        psf_row.addWidget(QLabel("Non-Stellar PSF radius (1–8):"))
        self.psf_spin = QDoubleSpinBox()
        self.psf_spin.setRange(1.0, 8.0); self.psf_spin.setSingleStep(0.5)
        self.psf_spin.setValue(3.0); self.psf_spin.setEnabled(False)
        psf_row.addWidget(self.psf_spin)
        vbox.addLayout(psf_row)

        self.sep_channels_chk = QCheckBox("Sharpen channels separately (colour only)")
        vbox.addWidget(self.sep_channels_chk)

        self.sharpen_stretch_chk = QCheckBox("Force pre-stretch (for linear / very dark images)")
        vbox.addWidget(self.sharpen_stretch_chk)

        vbox.addStretch()
        return w

    def _on_auto_psf_toggled(self, state):
        self.psf_spin.setEnabled(not self.auto_psf_chk.isChecked())

    def _tab_denoise(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(self._build_mode_model_box("denoise"))   # ← ADD
        vbox.addWidget(_label("Denoise Mode"))
        self.denoise_mode_combo = QComboBox()
        self.denoise_mode_combo.addItems(["full", "luminance", "separate"])
        vbox.addWidget(self.denoise_mode_combo)

        self.denoise_luma_slider, _  = _slider_row(vbox, "Luma Strength:", 0, 100, 50)
        self.denoise_color_slider, _ = _slider_row(vbox, "Colour Strength:", 0, 100, 50)

        vbox.addWidget(_label("Model Variant"))
        self.denoise_variant_combo = QComboBox()
        self.denoise_variant_combo.addItems([
            "Full",
            "Lite (faster, less VRAM)",
            "Walking Noise",
        ])
        vbox.addWidget(self.denoise_variant_combo)

        self.denoise_stretch_chk = QCheckBox("Force pre-stretch (for linear / very dark images)")
        vbox.addWidget(self.denoise_stretch_chk)

        vbox.addStretch()
        return w

    def _tab_superres(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(self._build_mode_model_box("superres"))  # ← ADD
        vbox.addWidget(_label("Scale Factor"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2×", "3×", "4×"])
        vbox.addWidget(self.scale_combo)

        vbox.addWidget(_label(
            "Super Resolution upscales the image using AI refinement.\n"
            "Bicubic upscaling is applied first, then AI enhances fine detail.",
            pt=9,
        ))
        vbox.addStretch()
        return w

    def _tab_darkstar(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setSpacing(10)
        vbox.addWidget(self._build_mode_model_box("darkstar"))  # ← ADD
        so_box  = QGroupBox("Stars-Only Output")
        so_vbox = QVBoxLayout(so_box)

        self.ds_stars_only_chk = QCheckBox("Compute stars-only layer")
        so_vbox.addWidget(self.ds_stars_only_chk)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Compositing mode:"))
        self.ds_mode_combo = QComboBox()
        self.ds_mode_combo.addItems(["unscreen", "additive"])
        self.ds_mode_combo.setToolTip(
            "unscreen: mathematically correct inverse of screen blending\n"
            "additive: simple difference (img − starless)"
        )
        mode_row.addWidget(self.ds_mode_combo)
        mode_row.addStretch()
        so_vbox.addLayout(mode_row)

        note = QLabel(
            "The stars-only result is saved as starmask_<filename> "
            "alongside the starless image."
        )
        note.setStyleSheet("QLabel { color: #777; font-style: italic; }")
        note.setWordWrap(True)
        so_vbox.addWidget(note)

        vbox.addWidget(so_box)
        vbox.addStretch()
        return w

    def _tab_satellite(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setSpacing(10)
        vbox.addWidget(self._build_mode_model_box("satellite")) # ← ADD
        mode_box  = QGroupBox("Processing Mode")
        mode_form = QVBoxLayout(mode_box)
        mode_row  = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.sat_mode_combo = QComboBox()
        self.sat_mode_combo.addItems(["full", "luminance"])
        self.sat_mode_combo.setToolTip(
            "full: detect and remove trails in all three channels\n"
            "luminance: process only the Y channel (faster, preserves colour)"
        )
        mode_row.addWidget(self.sat_mode_combo)
        mode_row.addStretch()
        mode_form.addLayout(mode_row)
        vbox.addWidget(mode_box)

        det_box  = QGroupBox("Detection & Removal")
        det_form = QVBoxLayout(det_box)

        self.sat_clip_chk = QCheckBox("Clip trail (recommended)")
        self.sat_clip_chk.setChecked(True)
        self.sat_clip_chk.stateChanged.connect(
            lambda s: self.sat_sens_slider.setEnabled(bool(s)))
        det_form.addWidget(self.sat_clip_chk)

        self.sat_sens_slider, _ = _slider_row(det_form, "Sensitivity:", 1, 100, 10)
        self.sat_sens_slider.setToolTip(
            "Lower = more aggressive trail clipping.\n"
            "Higher = only clip very bright trails.")

        self.sat_stretch_chk = QCheckBox("Force pre-stretch (for linear / very dark images)")
        self.sat_stretch_chk.setChecked(True)
        det_form.addWidget(self.sat_stretch_chk)
        vbox.addWidget(det_box)

        if not _HAS_TORCHVISION:
            warn = QLabel(
                "⚠  torchvision not found — satellite removal unavailable.\n"
                "It is normally installed alongside PyTorch; try reinstalling PyTorch."
            )
            warn.setStyleSheet("QLabel { color: #c00; font-style: italic; }")
            warn.setWordWrap(True)
            vbox.addWidget(warn)

        vbox.addStretch()
        return w

    def _tab_models(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setSpacing(10)

        mdl_box  = QGroupBox("Models")
        mdl_vbox = QVBoxLayout(mdl_box)

        self._settings_dir_lbl = QLabel()
        self._settings_dir_lbl.setWordWrap(True)
        mdl_vbox.addWidget(self._settings_dir_lbl)

    # Directory override
        dir_row = QHBoxLayout()
        self._models_dir_edit = QLineEdit()
        self._models_dir_edit.setPlaceholderText(
            "Default: Siril user-data / cosmic_clarity")
        self._models_dir_edit.setToolTip(
            "Leave blank to use the default location.\n"
            "Set to an existing models directory to use models already installed "
            "elsewhere (e.g. a SetiAstroSuite Pro installation) without "
            "downloading a second copy.")
        self._models_dir_edit.setReadOnly(True)
        dir_row.addWidget(self._models_dir_edit, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse_models_dir)
        dir_row.addWidget(browse_btn)

        clear_btn = QPushButton("Reset")
        clear_btn.setFixedWidth(60)
        clear_btn.setToolTip("Revert to the default models directory.")
        clear_btn.clicked.connect(self._on_reset_models_dir)
        dir_row.addWidget(clear_btn)
        mdl_vbox.addLayout(dir_row)

        # Populate from any existing override
        _ov = models_dir_override()
        if _ov:
            self._models_dir_edit.setText(str(_ov))

        dl_row = QHBoxLayout()
        self.dl_btn = QPushButton("Download Models (~2.4 GB)")
        self.dl_btn.setMinimumHeight(32)
        self.dl_btn.clicked.connect(self._on_download)
        dl_row.addWidget(self.dl_btn)

        self.dl_cancel_btn = QPushButton("Cancel Download")
        self.dl_cancel_btn.setEnabled(False)
        self.dl_cancel_btn.clicked.connect(self._on_cancel_download)
        dl_row.addWidget(self.dl_cancel_btn)
        mdl_vbox.addLayout(dl_row)

        self._advice_lbl = QLabel("If new AI4 models are released you may be able to download them "
            "separately without downloading the entire archive. Check the "
            "<a href='https://drive.google.com/drive/folders/1-fktZb3I9l-mQimJX2fZAmJCBj_t0yAF'>SetiAstro models Google drive</a>. "
            "Models must be unpacked into the location shown above.")
        self._advice_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._advice_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._advice_lbl.setOpenExternalLinks(True)
        self._advice_lbl.setWordWrap(True)
        mdl_vbox.addWidget(self._advice_lbl)

        self.dl_status_lbl = QLabel("")
        self.dl_status_lbl.setStyleSheet(
            "QLabel { color:#1976D2; font-style:italic; }")
        self.dl_status_lbl.setWordWrap(True)
        mdl_vbox.addWidget(self.dl_status_lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        mdl_vbox.addWidget(sep)

        prune_note = QLabel(
            "Remove files you no longer need to save disk space.\n"
            "If only one format is present, that backend will always be used."
        )
        prune_note.setStyleSheet("color:#555; font-style:italic;")
        prune_note.setWordWrap(True)
        mdl_vbox.addWidget(prune_note)

        prune_row = QHBoxLayout()
        self.prune_onnx_btn  = QPushButton("Remove all ONNX models")
        self.prune_torch_btn = QPushButton("Remove all Torch models")
        self.prune_obs_btn   = QPushButton("Remove obsolete models")
        self.prune_obs_btn.setToolTip(
            "Removes model files that are no longer used by this version of the "
            "script (old filenames from previous AI versions). Safe to run at any "
            "time — current model files are never touched.")
        for btn in (self.prune_onnx_btn, self.prune_torch_btn, self.prune_obs_btn):
            btn.setMinimumHeight(28)
            prune_row.addWidget(btn)
        self.prune_onnx_btn.clicked.connect(lambda: self._on_prune("onnx"))
        self.prune_torch_btn.clicked.connect(lambda: self._on_prune("torch"))
        self.prune_obs_btn.clicked.connect(self._on_prune_obsolete)
        mdl_vbox.addLayout(prune_row)

        self.prune_status_lbl = QLabel("")
        self.prune_status_lbl.setStyleSheet("color:#1976D2; font-style:italic;")
        self.prune_status_lbl.setWordWrap(True)
        mdl_vbox.addWidget(self.prune_status_lbl)

        ort_note = QLabel(
            "ONNX Runtime: "
            + ("available — ONNX models will be preferred on CPU/DirectML."
            if _get_ort() is not None
            else "not installed — Torch backend will be used for all models.\n"
                    "Install onnxruntime (or onnxruntime-directml on Windows) to enable.")
        )
        ort_note.setWordWrap(True)
        ort_note.setStyleSheet(
            "color:#2e7d32;" if _get_ort() else "color:#e65100; font-style:italic;")
        mdl_vbox.addWidget(ort_note)

        vbox.addWidget(mdl_box)

        # Initialise the directory label
        installed = any_models_installed()
        self._settings_dir_lbl.setText(
            f"Location: {models_dir()}"
            + ("" if installed else "\n⚠  No models found — please download.")
        )
        if not installed:
            self._settings_dir_lbl.setStyleSheet("color:#b71c1c;")

        vbox.addStretch()
        return w

    def _on_browse_models_dir(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        start = (self._models_dir_edit.text()
                or str(models_dir()))
        chosen = QFileDialog.getExistingDirectory(
            self, "Select models directory", start,
            QFileDialog.Option.ShowDirsOnly)
        if not chosen:
            return
        self._models_dir_edit.setText(chosen)
        set_models_dir_override(Path(chosen))
        self._refresh_model_status()
        self._sync_settings()


    def _on_reset_models_dir(self) -> None:
        self._models_dir_edit.clear()
        set_models_dir_override(None)
        self._refresh_model_status()
        self._sync_settings()

    def _on_prune_obsolete(self) -> None:
        self.prune_obs_btn.setEnabled(False)
        self.prune_onnx_btn.setEnabled(False)
        self.prune_torch_btn.setEnabled(False)
        self.prune_status_lbl.setText("Scanning for obsolete files…")

        self.prune_worker = ObsoleteWorker()
        self.prune_worker.log_signal.connect(
            lambda msg: self.prune_status_lbl.setText(msg))
        self.prune_worker.finished_signal.connect(self._on_prune_finished)
        self.prune_worker.start()

    def _on_prune_finished(self, ok: bool, msg: str) -> None:
        self.prune_onnx_btn.setEnabled(True)
        self.prune_torch_btn.setEnabled(True)
        self.prune_obs_btn.setEnabled(True)       # ← ADD
        self.prune_status_lbl.setText(msg)
        self._refresh_model_status()
        if not ok:
            QMessageBox.warning(self, "Prune failed", msg)

    def _tab_settings(self) -> QWidget:
        w    = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setSpacing(10)

        # Initialise the directory label now (before _refresh is wired up)
        installed = any_models_installed()
        self._settings_dir_lbl.setText(
            f"Location: {models_dir()}"
            + ("" if installed else "\n⚠  No models found — please download.")
        )
        if not installed:
            self._settings_dir_lbl.setStyleSheet("color:#b71c1c;")

        # --- Tiling -------------------------------------------------------
        tile_box  = QGroupBox("Tiling  (applies to all operations)")
        tile_form = QVBoxLayout(tile_box)

        note = QLabel(
            "Tile size and overlap must be multiples of 32.\n"
            "Larger tiles use more VRAM but may produce better blending."
        )
        note.setStyleSheet("QLabel { color:#555; font-style:italic; }")
        note.setWordWrap(True)
        tile_form.addWidget(note)

        def _spin32(lo, hi, default, label, tooltip=""):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(32)
            sp.setValue(default)
            if tooltip:
                sp.setToolTip(tooltip)
            sp.editingFinished.connect(
                lambda s=sp: s.setValue(round(s.value() / 32) * 32))
            row.addWidget(sp)
            row.addStretch()
            tile_form.addLayout(row)
            return sp

        self.tile_size_spin = _spin32(
            128, 1024, 512, "Tile size (px):",
            "Width and height of each processing tile.\n"
            "128–1024, must be a multiple of 32."
        )
        self.overlap_spin = _spin32(
            32, 96, 64, "Overlap (px):",
            "Overlap between adjacent tiles.\n"
            "32–96, must be a multiple of 32.\n"
            "Must be less than half the tile size."
        )

        def _validate_overlap():
            max_ov = (self.tile_size_spin.value() // 2) & ~31
            max_ov = max(32, min(max_ov, 96))
            self.overlap_spin.setMaximum(max_ov)
            if self.overlap_spin.value() > max_ov:
                self.overlap_spin.setValue(max_ov)

        self.tile_size_spin.valueChanged.connect(_validate_overlap)
        _validate_overlap()

        vbox.addWidget(tile_box)

        hw_box  = QGroupBox("Hardware")
        hw_vbox = QVBoxLayout(hw_box)
        self.gpu_chk = QCheckBox("Use GPU (CUDA / MPS / XPU / DirectML)")
        self.gpu_chk.setChecked(True)
        self.gpu_chk.stateChanged.connect(lambda _: self._refresh_model_status())
        hw_vbox.addWidget(self.gpu_chk)

        dev_lbl = QLabel(f"Detected device: {get_device(True)}")
        dev_lbl.setStyleSheet("QLabel { color: #555; font-style: italic; }")
        hw_vbox.addWidget(dev_lbl)

        # Backend preference                                          ← ADD
        bp_row = QHBoxLayout()
        bp_row.addWidget(QLabel("Inference backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(
            ["Automatic", "Prefer Torch", "Prefer ONNX Runtime"])
        self.backend_combo.setToolTip(
            "Automatic: ONNX Runtime on CPU/DirectML, Torch on CUDA/MPS/XPU.\n"
            "Prefer Torch: always use PyTorch if the .pth file is present.\n"
            "Prefer ONNX Runtime: always use ORT if the .onnx file is present.")
        self.backend_combo.currentIndexChanged.connect(
            self._on_backend_pref_changed)
        bp_row.addWidget(self.backend_combo)
        bp_row.addStretch()
        hw_vbox.addLayout(bp_row)

        vbox.addWidget(hw_box)
        vbox.addStretch()
        return w

    def _on_backend_pref_changed(self, idx: int) -> None:
        pref = ("auto", "torch", "onnx")[idx]
        set_backend_pref(pref)          # flushes _MODEL_CACHE
        self._refresh_model_status()

    def _build_mode_model_box(self, mode: str) -> QGroupBox:
        """
        Build a compact 'Models' status grid for one processing tab and register
        the label references so _refresh_model_status() can update them later.
        """
        from PyQt6.QtWidgets import QGridLayout

        box   = QGroupBox("Models")
        grid  = QGridLayout(box)
        grid.setSpacing(4)
        grid.setContentsMargins(8, 6, 8, 6)

        # Column headers
        for col, hdr in enumerate(_COL_HEADERS):
            lbl = QLabel(hdr)
            f = lbl.font(); f.setBold(True); lbl.setFont(f)
            lbl.setStyleSheet("color:#555;")
            grid.addWidget(lbl, 0, col)

        rows_info   = _MODE_MODEL_INFO.get(mode, [])
        label_refs  = []

        for r, (name, pth_key, onnx_key) in enumerate(rows_info, start=1):
            name_lbl  = QLabel(name)
            pth_lbl   = QLabel()
            onnx_lbl  = QLabel()
            pref_lbl  = QLabel()

            for lbl in (pth_lbl, onnx_lbl, pref_lbl):
                lbl.setMinimumWidth(68)

            grid.addWidget(name_lbl, r, 0)
            grid.addWidget(pth_lbl,  r, 1)
            grid.addWidget(onnx_lbl, r, 2)
            grid.addWidget(pref_lbl, r, 3)

            use_gpu = getattr(self, "gpu_chk", None)
            use_gpu = use_gpu.isChecked() if use_gpu else True
            _apply_row_state(pth_lbl, onnx_lbl, pref_lbl,
                            pth_key, onnx_key, use_gpu)

            label_refs.append((pth_lbl, onnx_lbl, pref_lbl, pth_key, onnx_key))

        self._mode_model_labels[mode] = label_refs
        return box

    def _refresh_model_status(self) -> None:
        """Re-evaluate all per-tab model indicators (called after download or GPU toggle)."""
        use_gpu = self.gpu_chk.isChecked()
        for mode, rows in self._mode_model_labels.items():
            for pth_lbl, onnx_lbl, pref_lbl, pth_key, onnx_key in rows:
                _apply_row_state(pth_lbl, onnx_lbl, pref_lbl,
                                pth_key, onnx_key, use_gpu)
        # Also refresh the settings-tab path label
        if hasattr(self, "_settings_dir_lbl"):
            installed = any_models_installed()
            self._settings_dir_lbl.setText(
                f"Location: {models_dir()}"
                + ("" if installed else "\n⚠  No models found — please download.")
            )
            self._settings_dir_lbl.setStyleSheet(
                "" if installed else "color:#b71c1c;")
        if hasattr(self, "_models_dir_edit"):
            ov = models_dir_override()
            if ov and not self._models_dir_edit.text():
                self._models_dir_edit.setText(str(ov))

    def _on_prune(self, target: str) -> None:
        label  = "ONNX" if target == "onnx" else "Torch (.pth / .pt)"
        answer = QMessageBox.question(
            self,
            f"Remove {label} models",
            f"This will permanently delete all {label} model files from:\n"
            f"{models_dir()}\n\n"
            "You can re-download them at any time. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.prune_onnx_btn.setEnabled(False)
        self.prune_torch_btn.setEnabled(False)
        self.prune_status_lbl.setText(f"Removing {label} files…")

        self.prune_worker = PruneWorker(target)
        self.prune_worker.log_signal.connect(
            lambda msg: self.prune_status_lbl.setText(msg))
        self.prune_worker.finished_signal.connect(self._on_prune_finished)
        self.prune_worker.start()


    def _on_prune_finished(self, ok: bool, msg: str) -> None:
        self.prune_onnx_btn.setEnabled(True)
        self.prune_torch_btn.setEnabled(True)
        self.prune_status_lbl.setText(msg)
        self._refresh_model_status()
        if not ok:
            QMessageBox.warning(self, "Prune failed", msg)

    def show_help(self) -> None:
        HelpDialog(self).exec()

    def _settings_path(self) -> Path:
        cfg_dir = Path(self.siril.get_siril_configdir()) / "CosmicClarity_Native"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "settings.json"


    def _save_settings(self) -> None:
        import json
        if getattr(self, "_settings_saved", False):
            return
        self._settings_saved = True
        self._sync_settings()   # last chance to read widgets (no-op if gone)
        if not self._settings_cache:
            return
        try:
            self._settings_path().write_text(
                json.dumps(self._settings_cache, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"CosmicClarity: could not save settings: {exc}")

    def _load_settings(self) -> None:
        import json
        try:
            raw = self._settings_path().read_text(encoding="utf-8")
            data: dict = json.loads(raw)
        except FileNotFoundError:
            return   # first run — use widget defaults
        except Exception as exc:
            print(f"CosmicClarity: could not load settings: {exc}")
            return

        def _set(widget, key, setter):
            if key in data:
                try:
                    setter(widget, data[key])
                except Exception:
                    pass

        _set(self.gpu_chk,              "use_gpu",            lambda w, v: w.setChecked(v))
        _set(self.tile_size_spin,       "tile_size",          lambda w, v: w.setValue(int(v)))
        _set(self.overlap_spin,         "overlap",            lambda w, v: w.setValue(int(v)))
        _set(self.backend_combo,        "backend_pref_idx",   lambda w, v: w.setCurrentIndex(int(v)))
        _set(self.sharpen_mode_combo,   "sharpen_mode",       lambda w, v: w.setCurrentText(v))
        _set(self.stellar_slider,       "stellar_amount",     lambda w, v: w.setValue(int(v)))
        _set(self.ns_slider,            "ns_amount",          lambda w, v: w.setValue(int(v)))
        _set(self.auto_psf_chk,         "auto_psf",           lambda w, v: w.setChecked(bool(v)))
        _set(self.psf_spin,             "psf_radius",         lambda w, v: w.setValue(float(v)))
        _set(self.sep_channels_chk,     "sep_channels",       lambda w, v: w.setChecked(bool(v)))
        _set(self.sharpen_stretch_chk,  "sharpen_stretch",    lambda w, v: w.setChecked(bool(v)))
        _set(self.denoise_mode_combo,   "denoise_mode",       lambda w, v: w.setCurrentText(v))
        _set(self.denoise_luma_slider,  "denoise_luma",       lambda w, v: w.setValue(int(v)))
        _set(self.denoise_color_slider, "denoise_color",      lambda w, v: w.setValue(int(v)))
        _set(self.denoise_variant_combo, "denoise_variant",    lambda w, v: w.setCurrentText(str(v)))
        # Backward compat: upgrade old boolean denoise_lite key
        if "denoise_lite" in data and data.get("denoise_lite") and "denoise_variant" not in data:
            self.denoise_variant_combo.setCurrentText("Lite (faster, less VRAM)")
        _set(self.denoise_stretch_chk,  "denoise_stretch",    lambda w, v: w.setChecked(bool(v)))
        _set(self.scale_combo,          "superres_scale_idx", lambda w, v: w.setCurrentIndex(int(v)))
        _set(self.ds_stars_only_chk,    "darkstar_stars_only",lambda w, v: w.setChecked(bool(v)))
        _set(self.ds_mode_combo,        "darkstar_mode",      lambda w, v: w.setCurrentText(v))
        _set(self.sat_mode_combo,       "sat_mode",           lambda w, v: w.setCurrentText(v))
        _set(self.sat_clip_chk,         "sat_clip",           lambda w, v: w.setChecked(bool(v)))
        _set(self.sat_sens_slider,      "sat_sensitivity",    lambda w, v: w.setValue(int(v)))
        _set(self.sat_stretch_chk,      "sat_stretch",        lambda w, v: w.setChecked(bool(v)))
        # Restore models directory override before refreshing status
        ov_str = data.get("models_dir_override", "")
        if ov_str and Path(ov_str).is_dir():
            set_models_dir_override(Path(ov_str))
            if hasattr(self, "_models_dir_edit"):
                self._models_dir_edit.setText(ov_str)
        else:
            set_models_dir_override(None)
        # Sync the backend global to whatever was restored
        self._on_backend_pref_changed(self.backend_combo.currentIndex())

    # ------------------------------------------------------------------ run / cancel

    def _collect_params(self) -> dict:
        tab = self.tabs.currentIndex()
        # Settings (tab 5) always provide tile size, overlap and GPU flag
        base = {
            "use_gpu":    self.gpu_chk.isChecked(),
            "chunk_size": self.tile_size_spin.value(),
            "overlap":    self.overlap_spin.value(),
        }

        if tab == 0:  # Sharpen
            base.update({
                "sharpening_mode":             self.sharpen_mode_combo.currentText(),
                "stellar_amount":              self.stellar_slider.value() / 100.0,
                "nonstellar_amount":           self.ns_slider.value() / 100.0,
                "auto_psf":                    self.auto_psf_chk.isChecked(),
                "nonstellar_psf":              self.psf_spin.value(),
                "sharpen_channels_separately": self.sep_channels_chk.isChecked(),
                "temp_stretch":                self.sharpen_stretch_chk.isChecked(),
                "target_median":               0.25,
            })
        elif tab == 1:  # Denoise
            base.update({
                "denoise_mode":     self.denoise_mode_combo.currentText(),
                "denoise_strength": self.denoise_luma_slider.value() / 100.0,
                "denoise_color":    self.denoise_color_slider.value() / 100.0,
                "lite":             self.denoise_variant_combo.currentText() == "Lite (faster, less VRAM)",
                "walking":          self.denoise_variant_combo.currentText() == "Walking Noise",
                "temp_stretch":     self.denoise_stretch_chk.isChecked(),
                "target_median":    0.25,
            })
        elif tab == 2:  # Super-res
            base.update({
                "scale": int(self.scale_combo.currentText()[0]),
            })
        elif tab == 3:  # DarkStar
            base.update({
                "darkstar_mode":    self.ds_mode_combo.currentText(),
                "output_stars_only":self.ds_stars_only_chk.isChecked(),
            })
        elif tab == 4:  # Satellite
            base.update({
                "sat_mode":     self.sat_mode_combo.currentText(),
                "clip_trail":   self.sat_clip_chk.isChecked(),
                "sensitivity":  self.sat_sens_slider.value() / 100.0,
                "temp_stretch": self.sat_stretch_chk.isChecked(),
                "target_median": 0.25,
            })
        return base

    def _current_mode(self) -> str:
        tab = self.tabs.currentIndex()
        if tab == 0: return "sharpen"
        if tab == 1:
            v = self.denoise_variant_combo.currentText()
            if v == "Lite (faster, less VRAM)": return "denoise_lite"
            if v == "Walking Noise":            return "denoise_1w"
            return "denoise"
        if tab == 2: return "superres"
        if tab == 3: return "darkstar"
        return "satellite"

    def _on_run_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.run_btn.setEnabled(False)
            self.progress_label.setText("Cancelling…")
            return

        mode = self._current_mode()
        req  = MODE_MODELS.get(mode, [])
        if req and not all(model_path(k).exists() for k in req):
            QMessageBox.critical(self, "Models missing",
                                 f"Required models for '{mode}' are not installed.\n"
                                 "Please download the models first.")
            return

        params = self._collect_params()
        self.worker = ProcessWorker(mode, params, self.siril)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

        self.run_btn.setText("Cancel")
        self.tabs.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")

    # ------------------------------------------------------------------ download

    def _on_download(self):
        self.dl_btn.setEnabled(False)
        self.dl_cancel_btn.setEnabled(True)
        self.dl_status_lbl.setText("Starting download…")

        zip_dst = models_dir() / "cc_models.zip"
        self.dl_worker = DownloadWorker(zip_dst)
        self.dl_worker.log_signal.connect(
            lambda msg: self.dl_status_lbl.setText(msg))
        self.dl_worker.finished_signal.connect(self._on_dl_finished)
        self.dl_worker.start()

    def _on_cancel_download(self):
        if self.dl_worker:
            self.dl_worker.cancel()
        self.dl_cancel_btn.setEnabled(False)
        self.dl_status_lbl.setText("Cancelling download…")

    def _on_dl_finished(self, ok: bool, msg: str):
        self.dl_btn.setEnabled(True)
        self.dl_cancel_btn.setEnabled(False)
        self.dl_status_lbl.setText(msg)
        if ok:
            self._refresh_model_status()          # ← replaces the old single-label update
            QMessageBox.information(self, "Download complete", msg)
        else:
            QMessageBox.warning(self, "Download failed", msg)

    # ------------------------------------------------------------------ progress / result

    def _on_progress(self, done: int, total: int, label: str):
        pct = int(done * 100 / max(total, 1))
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"{label}  ({done}/{total})")
        self.siril.update_progress(label, pct / 100.0)

    def _on_log(self, msg: str):
        self.progress_label.setText(msg)

    def _on_finished(self, ok: bool, msg: str):
        self.run_btn.setText("Run")
        self.run_btn.setEnabled(True)
        self.tabs.setEnabled(True)
        self.siril.reset_progress()

        if ok:
            self.progress_bar.setValue(100)
            self.progress_label.setText("Complete.")
        else:
            self.progress_label.setText(f"{'Cancelled.' if 'cancel' in msg.lower() else 'Error: ' + msg[:120]}")
            if "cancel" not in msg.lower():
                QMessageBox.critical(self, "Processing error", msg)

    def show_about(self) -> None:
        """Show the instructions dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

def closeEvent(self, event):
    self._save_settings()
    if self.worker and self.worker.isRunning():
        self.worker.cancel()
        self.worker.wait(3000)
    if self.dl_worker and self.dl_worker.isRunning():
        self.dl_worker.cancel()
        self.dl_worker.wait(3000)
    if self.prune_worker and self.prune_worker.isRunning():
        self.prune_worker.wait(3000)
    self.siril.disconnect()
    event.accept()

# =============================================================================
# CLI / headless mode
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cosmic_clarity_siril.py",
        description="Cosmic Clarity AI4 for Siril — CLI mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=["sharpen", "denoise", "superres", "darkstar", "satellite"],
                   help="Processing mode")

    # Hardware
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU (disable GPU)")
    p.add_argument("--tile-size", type=int, default=512, metavar="PIXELS",
                   help="Tile size for all operations (128–1024, multiple of 32)")
    p.add_argument("--overlap",   type=int, default=64,  metavar="PIXELS",
                   help="Tile overlap for all operations (32–96, multiple of 32)")

    # Sharpen
    sg = p.add_argument_group("Sharpen options")
    sg.add_argument("--sharpening-mode", default="Both",
                    choices=["Both", "Stellar Only", "Non-Stellar Only"])
    sg.add_argument("--stellar-amount",     type=float, default=0.5,
                    metavar="0-1")
    sg.add_argument("--nonstellar-amount",  type=float, default=0.5,
                    metavar="0-1")
    sg.add_argument("--nonstellar-psf",     type=float, default=3.0,
                    metavar="1-8",
                    help="PSF radius when auto-PSF is disabled")
    sg.add_argument("--no-auto-psf",        action="store_true",
                    help="Disable per-chunk PSF auto-measurement")
    sg.add_argument("--sharpen-channels",   action="store_true",
                    help="Sharpen R/G/B channels separately")
    sg.add_argument("--sharpen-stretch",    action="store_true",
                    help="Force pre-stretch before sharpening")

    # Denoise
    dg = p.add_argument_group("Denoise options")
    dg.add_argument("--denoise-mode",     default="full",
                    choices=["full", "luminance", "separate"])
    dg.add_argument("--denoise-strength", type=float, default=0.5,
                    metavar="0-1")
    dg.add_argument("--denoise-color",    type=float, default=None,
                    metavar="0-1",
                    help="Colour denoising strength (defaults to --denoise-strength)")
    dg.add_argument("--lite",             action="store_true",
                    help="Use the lighter NAFNet variant")
    dg.add_argument("--walking",          action="store_true",
                    help="Use the Walking Noise NAFNet variant")
    dg.add_argument("--denoise-stretch",  action="store_true",
                    help="Force pre-stretch before denoising")

    # Super-res
    srg = p.add_argument_group("Super-resolution options")
    srg.add_argument("--scale", type=int, default=2,
                     choices=[2, 3, 4])

    # DarkStar
    dsg = p.add_argument_group("Star removal options")
    dsg.add_argument("--darkstar-mode",   default="unscreen",
                     choices=["unscreen", "additive"])
    dsg.add_argument("--stars-only",      action="store_true",
                     help="Compute and save stars-only layer")

    # Satellite
    satg = p.add_argument_group("Satellite removal options")
    satg.add_argument("--sat-mode",      default="full", choices=["full", "luminance"])
    satg.add_argument("--no-clip-trail", action="store_true",
                      help="Disable trail clipping (keep raw network output)")
    satg.add_argument("--sensitivity",   type=float, default=0.1, metavar="0-1",
                      help="Trail clip sensitivity (lower = more aggressive)")
    satg.add_argument("--sat-stretch",   action="store_true",
                      help="Force pre-stretch before satellite removal")

    return p


def _cli_progress(done: int, total: int, label: str) -> None:
    pct = int(done * 100 / max(total, 1))
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {label}          ", end="", flush=True)
    if done >= total:
        print()


def run_cli(argv: list[str]) -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    # Connect to Siril
    try:
        siril = s.SirilInterface()
        siril.connect()
    except Exception as exc:
        print(f"ERROR: Could not connect to Siril: {exc}", file=sys.stderr)
        return 1

    set_siril_iface(siril)   # ← add this line

    try:
        try:
            siril.cmd("requires", "1.4.2")
        except Exception as exc:
            print(f"ERROR: Siril version requirement not met: {exc}", file=sys.stderr)
            return 1

        if not siril.is_image_loaded():
            print("ERROR: No image is loaded in Siril.", file=sys.stderr)
            return 1

        init_models_dir(siril)

        mode = args.mode
        _denoise_mode = ("denoise_1w" if args.walking else
                         "denoise_lite" if args.lite else mode)
        req  = MODE_MODELS.get(_denoise_mode if mode == "denoise" else mode,
                               MODE_MODELS.get(mode, []))
        missing = [k for k in req if not model_path(k).exists()]
        if missing:
            print(f"ERROR: Required models missing: {missing}", file=sys.stderr)
            print(f"       Expected location: {models_dir()}", file=sys.stderr)
            print("       Run the script in GUI mode and click 'Download Models'.",
                  file=sys.stderr)
            return 1

        # Snap tile-size and overlap to nearest multiple of 32 and validate
        tile_size = max(128, min(1024, (args.tile_size // 32) * 32))
        overlap   = max(32,  min(96,   (args.overlap   // 32) * 32))
        overlap   = min(overlap, (tile_size // 2) & ~31)  # must be < tile_size/2
        overlap   = max(32, overlap)

        params: dict = {
            "use_gpu":    not args.cpu,
            "chunk_size": tile_size,
            "overlap":    overlap,
        }

        if mode == "sharpen":
            params.update({
                "sharpening_mode":             args.sharpening_mode,
                "stellar_amount":              args.stellar_amount,
                "nonstellar_amount":           args.nonstellar_amount,
                "nonstellar_psf":              args.nonstellar_psf,
                "auto_psf":                    not args.no_auto_psf,
                "sharpen_channels_separately": args.sharpen_channels,
                "temp_stretch":                args.sharpen_stretch,
                "target_median":               0.25,
            })
        elif mode == "denoise":
            params.update({
                "denoise_mode":    args.denoise_mode,
                "denoise_strength":args.denoise_strength,
                "denoise_color":   args.denoise_color if args.denoise_color is not None
                                   else args.denoise_strength,
                "lite":            args.lite,
                "walking":         args.walking,
                "temp_stretch":    args.denoise_stretch,
                "target_median":   0.25,
            })
        elif mode == "superres":
            params.update({"scale": args.scale})
        elif mode == "darkstar":
            params.update({
                "darkstar_mode":    args.darkstar_mode,
                "output_stars_only":args.stars_only,
            })
        elif mode == "satellite":
            params.update({
                "sat_mode":      args.sat_mode,
                "clip_trail":    not args.no_clip_trail,
                "sensitivity":   args.sensitivity,
                "temp_stretch":  args.sat_stretch,
                "target_median": 0.25,
            })

        img, was_mono, was_planar, is_uint16 = siril_get_image(siril)
        print(f"Image: {img.shape[1]}×{img.shape[0]} px  "
                f"{'mono' if was_mono else 'colour'}  "
                f"{'uint16' if is_uint16 else 'float32'}")

            # siril.undo_save_state(f"Cosmic Clarity – {mode}")
            # (currently we don't save undo states for commands)

        print(f"Running: {mode} …")
        if mode == "sharpen":
            result = process_sharpen(img, params, progress_cb=_cli_progress)
            siril_set_image(siril, result, was_mono, was_planar, is_uint16)

        elif mode == "denoise":
            result = process_denoise(img, params, progress_cb=_cli_progress)
            siril_set_image(siril, result, was_mono, was_planar, is_uint16)

        elif mode == "superres":
            result = process_superres(img, params, progress_cb=_cli_progress)
            # result is (H*scale, W*scale, 3) — sirilpy expects (C, H, W)
            out = np.transpose(result.astype(np.float32, copy=False), (2, 0, 1))
            with siril.image_lock():
                siril.set_image_pixeldata(out)

        elif mode == "darkstar":
            starless, so = process_darkstar(img, params, progress_cb=_cli_progress)
            siril_set_image(siril, starless, was_mono, was_planar, is_uint16)
            if so is not None:
                image = self.siril.get_image(with_pixels=False) # Get metadata
                # Update filter metadata and save the image
                lines = image.header.split('\n')
                lines = [line for line in lines if not line.strip().startswith('END')]
                while lines and not lines[-1].strip():
                    lines.pop()
                header_dict = fits.Header.fromstring('\n'.join(lines), sep='\n')
                header_dict["FILTER"] = "starless"
                hdu = fits.PrimaryHDU(header=header_dict)
                hdu.header.remove('BG-PTS', ignore_missing=True) # Remove broken GraXpert header card if present
                hdu.verify('silentfix')
                header_str = hdu.header.tostring(sep='\n')
                orig_filename = siril.get_image_filename()
                savename = "starmask_" + Path(orig_filename).name
                if was_mono:
                    so = so[..., 0]          # back to 2-D

                if was_planar and so.ndim == 3:
                    so = np.transpose(so, (2, 0, 1))

                if is_uint16:
                    so = (np.clip(so, 0, 1) * 65535.0).astype(np.uint16)
                siril.save_image_file(so, header=header_str, filename=savename)
                print(
                    f"Starmask layer computed and saved as {savename}."
                )

        elif mode == "satellite":
            result, detected = process_satellite(img, params, progress_cb=_cli_progress)
            siril_set_image(siril, result, was_mono, was_planar, is_uint16)
            if not detected:
                print("No satellite trails detected in this image.")

        siril.log(f"Cosmic Clarity [{mode}] complete.")
        print("Done.")
        return 0

    finally:
        siril.disconnect()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    # Detect CLI mode: any argument that isn't a Siril/Qt internal flag
    cli_argv = [a for a in sys.argv[1:]
                if not a.startswith("-psn") and not a.startswith("--siril")]

    if cli_argv and cli_argv[0] not in ("--help", "-h", "--version"):
        # Headless / CLI path
        sys.exit(run_cli(cli_argv))

    elif "--help" in cli_argv or "-h" in cli_argv:
        _build_arg_parser().print_help()
        sys.exit(0)

    # --- GUI path ---
    try:
        siril = s.SirilInterface()
        siril.connect()
    except s.SirilConnectionError:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Connection error",
                             "Could not connect to Siril.\n"
                             "Please run this script from within Siril.")
        sys.exit(1)

    try:
        siril.cmd("requires", "1.4.2")
    except Exception as exc:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Version error", str(exc))
        siril.disconnect()
        sys.exit(1)

    if not siril.is_image_loaded():
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "No image",
                             "No image is currently loaded in Siril.\n"
                             "Please open an image before running Cosmic Clarity.")
        siril.disconnect()
        sys.exit(1)

    set_siril_iface(siril)
    init_models_dir(siril)

    app = QApplication.instance() or QApplication(sys.argv)
    gui = CosmicClarityGUI(siril)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

