from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import traceback
import types
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from image_metrics import (
    _box_blur_gray,
    _clamp_float,
    _to_rgb_float_fullres,
    _to_rgb_float_image,
    measure_image_features,
    measure_quality_metrics,
)
from models import ImageFeatures, QualityMetrics

ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

try:
    from sirilpy.exceptions import CommandError, SirilError
except Exception:  # Tests may import with lightweight fakes.
    CommandError = Exception
    SirilError = Exception

def find_latest_sasp_wheel(pipeline) -> Optional[Path]:
    if not pipeline.siril_plugin_dir:
        return None
    downloads_dir = pipeline.siril_plugin_dir / "downloads"
    if not downloads_dir.exists() or not downloads_dir.is_dir():
        return None
    wheels = sorted(
        downloads_dir.glob("setiastrosuitepro-*.whl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return wheels[0] if wheels else None

def install_pyqt6_headless_stub(pipeline) -> bool:
    if "PyQt6" in sys.modules:
        return False

    class _DummySignal:
        def connect(self, *_args, **_kwargs):
            return None

        def emit(self, *_args, **_kwargs):
            return None

    def _dummy_callable(*_args, **_kwargs):
        return None

    class _DummyQtObject:
        def __init__(self, *_args, **_kwargs):
            pass

        def __getattr__(self, _name):
            return _dummy_callable

    class _QThread(_DummyQtObject):
        pass

    class _QStandardPaths:
        class StandardLocation:
            AppDataLocation = 0

        @staticmethod
        def writableLocation(_loc):
            return os.path.expanduser(
                "~/Library/Application Support/SeestarSuperimpose/runtime_home/setiastro"
            )

    class _QSettings:
        _store: Dict[str, Any] = {}

        def value(self, key, default=None, type=None):  # noqa: A002
            value = pipeline._store.get(key, default)
            if type is bool:
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in ("1", "true", "yes", "on"):
                        return True
                    if lowered in ("0", "false", "no", "off"):
                        return False
                return bool(value)
            if type is str:
                if value is None:
                    return ""
                return str(value)
            if type is None:
                return value
            try:
                return type(value)
            except Exception:
                return value

        def setValue(self, key, value):
            pipeline._store[key] = value

    class _QMessageBox(_DummyQtObject):
        @staticmethod
        def critical(*_args, **_kwargs):
            return None

        @staticmethod
        def warning(*_args, **_kwargs):
            return None

        @staticmethod
        def information(*_args, **_kwargs):
            return None

    qt_mod = types.ModuleType("PyQt6")
    qt_core = types.ModuleType("PyQt6.QtCore")
    qt_widgets = types.ModuleType("PyQt6.QtWidgets")
    qt_gui = types.ModuleType("PyQt6.QtGui")
    qt_quick_widgets = types.ModuleType("PyQt6.QtQuickWidgets")

    qt_core.Qt = types.SimpleNamespace(
        AlignmentFlag=types.SimpleNamespace(AlignCenter=0),
        TextFormat=types.SimpleNamespace(RichText=0),
        TextInteractionFlag=types.SimpleNamespace(TextBrowserInteraction=0),
        KeyboardModifier=types.SimpleNamespace(ControlModifier=0),
        Orientation=types.SimpleNamespace(Horizontal=0, Vertical=1),
        WidgetAttribute=types.SimpleNamespace(WA_DontShowOnScreen=0, WA_DeleteOnClose=1),
        WindowType=types.SimpleNamespace(Window=0),
        AspectRatioMode=types.SimpleNamespace(KeepAspectRatio=0),
        TransformationMode=types.SimpleNamespace(SmoothTransformation=0),
    )
    qt_core.QObject = _DummyQtObject
    qt_core.QThread = _QThread
    qt_core.QTimer = type("QTimer", (_DummyQtObject,), {})
    qt_core.QPoint = type("QPoint", (_DummyQtObject,), {})
    qt_core.QPointF = type("QPointF", (_DummyQtObject,), {})
    qt_core.QEvent = type("QEvent", (_DummyQtObject,), {})
    qt_core.QUrl = type("QUrl", (_DummyQtObject,), {})
    qt_core.QByteArray = type("QByteArray", (_DummyQtObject,), {})
    qt_core.pyqtSignal = lambda *_a, **_k: _DummySignal()
    qt_core.pyqtProperty = lambda *_a, **_k: property(lambda _self: None)
    qt_core.QCoreApplication = type("QCoreApplication", (_DummyQtObject,), {})
    qt_core.QStandardPaths = _QStandardPaths
    qt_core.QSettings = _QSettings

    for name in (
        "QDialog",
        "QVBoxLayout",
        "QHBoxLayout",
        "QLabel",
        "QPushButton",
        "QFileDialog",
        "QComboBox",
        "QSpinBox",
        "QProgressBar",
        "QCheckBox",
        "QLineEdit",
        "QApplication",
        "QMainWindow",
        "QWidget",
        "QGroupBox",
        "QFormLayout",
        "QFrame",
        "QSlider",
        "QToolButton",
        "QScrollArea",
        "QGraphicsView",
        "QGraphicsScene",
        "QGraphicsPixmapItem",
        "QProgressBar",
        "QMdiArea",
    ):
        setattr(qt_widgets, name, type(name, (_DummyQtObject,), {}))
    qt_widgets.QMessageBox = _QMessageBox

    qt_gui.QAction = type("QAction", (_DummyQtObject,), {})
    qt_gui.QIcon = type("QIcon", (_DummyQtObject,), {})
    qt_gui.QFont = type("QFont", (_DummyQtObject,), {})
    qt_gui.QImage = type("QImage", (_DummyQtObject,), {})
    qt_gui.QPixmap = type("QPixmap", (_DummyQtObject,), {})
    qt_gui.QMovie = type("QMovie", (_DummyQtObject,), {})
    qt_gui.QWheelEvent = type("QWheelEvent", (_DummyQtObject,), {})
    qt_gui.QIntValidator = type("QIntValidator", (_DummyQtObject,), {})
    qt_gui.QDoubleValidator = type("QDoubleValidator", (_DummyQtObject,), {})
    qt_gui.QPainter = type("QPainter", (_DummyQtObject,), {
        "RenderHint": types.SimpleNamespace(
            SmoothPixmapTransform=0,
            Antialiasing=1,
        )
    })
    qt_quick_widgets.QQuickWidget = type("QQuickWidget", (_DummyQtObject,), {})

    qt_mod.QtCore = qt_core
    qt_mod.QtWidgets = qt_widgets
    qt_mod.QtGui = qt_gui
    qt_mod.QtQuickWidgets = qt_quick_widgets

    sys.modules["PyQt6"] = qt_mod
    sys.modules["PyQt6.QtCore"] = qt_core
    sys.modules["PyQt6.QtWidgets"] = qt_widgets
    sys.modules["PyQt6.QtGui"] = qt_gui
    sys.modules["PyQt6.QtQuickWidgets"] = qt_quick_widgets
    return True

def load_sasp_aberration_module(pipeline):
    if pipeline._sasp_aberration_module is not None:
        return pipeline._sasp_aberration_module
    if pipeline._sasp_aberration_module_error is not None:
        return None

    wheel_path = pipeline._find_latest_sasp_wheel()
    if wheel_path is None:
        pipeline._sasp_aberration_module_error = (
            "setiastrosuitepro wheel not found in plugin downloads"
        )
        return None

    wheel_token = str(wheel_path)
    if wheel_token not in sys.path:
        sys.path.insert(0, wheel_token)

    try:
        module = importlib.import_module("setiastro.saspro.aberration_ai")
    except Exception as e:
        if "PyQt6" in str(e):
            stubbed = pipeline._install_pyqt6_headless_stub()
            if stubbed:
                pipeline.log.warn(
                    "SASP Aberration API: PyQt6 missing, use headless stub fallback"
                )
                sys.modules.pop("setiastro.saspro.aberration_ai", None)
                try:
                    module = importlib.import_module("setiastro.saspro.aberration_ai")
                except Exception as e2:
                    pipeline._sasp_aberration_module_error = (
                        f"import failed after PyQt6 stub: {pipeline._short_text(e2)}"
                    )
                    pipeline.log.warn(
                        f"SASP Aberration API import failed after stub: {e2}"
                    )
                    if pipeline.cfg.debug_mode:
                        pipeline.log.debug(traceback.format_exc())
                    return None
            else:
                pipeline._sasp_aberration_module_error = (
                    f"import failed: {pipeline._short_text(e)}"
                )
                pipeline.log.warn(f"SASP Aberration API import failed: {e}")
                if pipeline.cfg.debug_mode:
                    pipeline.log.debug(traceback.format_exc())
                return None
        else:
            pipeline._sasp_aberration_module_error = (
                f"import failed: {pipeline._short_text(e)}"
            )
            pipeline.log.warn(f"SASP Aberration API import failed: {e}")
            if pipeline.cfg.debug_mode:
                pipeline.log.debug(traceback.format_exc())
            return None

    runner = getattr(module, "run_aberration_ai_on_array", None)
    if not callable(runner):
        pipeline._sasp_aberration_module_error = (
            "run_aberration_ai_on_array is missing"
        )
        pipeline.log.warn("SASP Aberration API unavailable: missing runner function")
        return None

    pipeline._sasp_aberration_module = module
    pipeline._sasp_aberration_module_error = None
    return module

def prepare_aberration_input(pipeline, image_data):
    arr = np.asarray(image_data)
    dtype = arr.dtype
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False), "mono", dtype
    if arr.ndim != 3:
        raise ValueError(f"unsupported image shape: {arr.shape}")

    if arr.shape[0] in (1, 3):
        hwc = np.transpose(arr, (1, 2, 0))
        return hwc.astype(np.float32, copy=False), "chw", dtype
    if arr.shape[2] in (1, 3):
        return arr.astype(np.float32, copy=False), "hwc", dtype
    raise ValueError(f"unsupported channel layout: {arr.shape}")

def restore_aberration_output(pipeline, output_data, layout: str, src_dtype):
    out = np.asarray(output_data)

    if layout == "mono":
        if out.ndim == 3:
            if out.shape[2] not in (1, 3):
                raise ValueError(f"unexpected Aberration output shape: {out.shape}")
            out = out[:, :, 0]
        elif out.ndim != 2:
            raise ValueError(f"unexpected Aberration output shape: {out.shape}")
    elif layout == "chw":
        if out.ndim == 2:
            out = out[:, :, None]
        if out.ndim != 3 or out.shape[2] not in (1, 3):
            raise ValueError(f"unexpected Aberration output shape: {out.shape}")
        out = np.transpose(out, (2, 0, 1))
    elif layout == "hwc":
        if out.ndim == 2:
            out = out[:, :, None]
        if out.ndim != 3 or out.shape[2] not in (1, 3):
            raise ValueError(f"unexpected Aberration output shape: {out.shape}")
    else:
        raise ValueError(f"unknown layout: {layout}")

    if np.issubdtype(src_dtype, np.integer):
        if src_dtype == np.uint16:
            out = np.clip(out, 0, 65535)
        out = out.astype(src_dtype, copy=False)
    else:
        out = out.astype(np.float32, copy=False)
    return out

def resolve_local_aberration_model(pipeline) -> Optional[Path]:
    if not pipeline.siril_plugin_dir:
        return None

    direct_candidates = [
        pipeline.siril_plugin_dir / "model_v2_0_1.onnx",
        pipeline.siril_plugin_dir / "downloads" / "model_v2_0_1.onnx",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    downloads_dir = pipeline.siril_plugin_dir / "downloads"
    if downloads_dir.is_dir():
        for candidate in sorted(
            downloads_dir.glob("*.onnx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if candidate.is_file():
                return candidate
    return None

def preferred_aberration_providers(pipeline, module) -> Tuple[Optional[List[str]], str]:
    provider_mode = os.getenv("SEESTAR_ABERRATION_PROVIDER", "").strip().lower()
    ort_mod = getattr(module, "ort", None)
    if ort_mod is None:
        return None, "default"

    try:
        available = list(ort_mod.get_available_providers())
    except Exception:
        return None, "default"

    available_set = set(available)
    if provider_mode in ENV_FALSE_VALUES or provider_mode == "cpu":
        if "CPUExecutionProvider" in available_set:
            return ["CPUExecutionProvider"], "cpu-forced"
        return None, "cpu-forced-unavailable"

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if "CoreMLExecutionProvider" in available_set:
            providers = ["CoreMLExecutionProvider"]
            if "CPUExecutionProvider" in available_set:
                providers.append("CPUExecutionProvider")
            return providers, "coreml-preferred"

    if provider_mode in ENV_TRUE_VALUES or provider_mode == "coreml":
        if "CoreMLExecutionProvider" in available_set:
            providers = ["CoreMLExecutionProvider"]
            if "CPUExecutionProvider" in available_set:
                providers.append("CPUExecutionProvider")
            return providers, "coreml-forced"

    return None, "default"

def run_aberration_api(pipeline, step_key: str, model_path: Optional[Path] = None):
    pipeline._last_aberration_api_error = None
    module = pipeline._load_sasp_aberration_module()
    if module is None:
        reason = pipeline._sasp_aberration_module_error or "module unavailable"
        pipeline._last_aberration_api_error = reason
        pipeline.log.warn(f"{step_key} Aberration API unavailable: {reason}")
        return None

    resolved_model_path = model_path if model_path and model_path.is_file() else None
    if model_path is not None and resolved_model_path is None:
        pipeline.log.warn(f"{step_key} Aberration API local model not found: {model_path}")

    def _log_cb(msg):
        text = str(msg).strip()
        if text:
            pipeline.log.info(f"[{step_key}] {text}")

    def _run_with_pixels() -> Tuple[np.ndarray, str]:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("image buffer is empty")
        input_image, input_layout, input_dtype = pipeline._prepare_aberration_input(image_data)
        preferred_providers, provider_strategy = pipeline._preferred_aberration_providers(module)
        run_kwargs: Dict[str, Any] = {
            "model_path": str(resolved_model_path) if resolved_model_path else None,
            "auto_gpu": True,
            "log_cb": _log_cb,
        }
        if preferred_providers:
            run_kwargs["providers"] = preferred_providers
            if provider_strategy.startswith("coreml"):
                _log_cb(
                    "Aberration provider override enabled: "
                    + " -> ".join(preferred_providers)
                )
        output_image = None
        provider = None
        orig_is_apple_arm = getattr(module, "IS_APPLE_ARM", None)
        orig_model_required_patch = getattr(module, "_model_required_patch", None)
        try:
            if preferred_providers and provider_strategy.startswith("coreml"):
                if hasattr(module, "IS_APPLE_ARM"):
                    module.IS_APPLE_ARM = False
                if callable(orig_model_required_patch):
                    module._model_required_patch = lambda _path: 0
            output_image, provider = module.run_aberration_ai_on_array(
                input_image,
                **run_kwargs,
            )
        except Exception as coreml_exc:
            if not (preferred_providers and provider_strategy.startswith("coreml")):
                raise
            _log_cb(
                "Aberration provider override failed; retrying default path: "
                f"{pipeline._short_text(coreml_exc, 180)}"
            )
            output_image, provider = module.run_aberration_ai_on_array(
                input_image,
                model_path=str(resolved_model_path) if resolved_model_path else None,
                auto_gpu=True,
                log_cb=_log_cb,
            )
        finally:
            if hasattr(module, "IS_APPLE_ARM") and orig_is_apple_arm is not None:
                module.IS_APPLE_ARM = orig_is_apple_arm
            if callable(orig_model_required_patch):
                module._model_required_patch = orig_model_required_patch
        restored_image = pipeline._restore_aberration_output(
            output_image,
            input_layout,
            input_dtype,
        )
        pipeline.siril.set_image_pixeldata(restored_image)
        provider_name = str(provider).strip() if provider is not None else ""
        return restored_image, provider_name

    try:
        lock_factory = getattr(pipeline.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                _restored, provider_name = _run_with_pixels()
        else:
            pipeline.log.warn(
                f"{step_key} Aberration API: image_lock unavailable, running without thread lock"
            )
            _restored, provider_name = _run_with_pixels()

        label = (
            f"SASP Aberration API ({provider_name})"
            if provider_name else "SASP Aberration API"
        )
        if resolved_model_path:
            label += f" [{resolved_model_path.name}]"
        pipeline.workflow_command_used[step_key] = label
        pipeline.log.info(f"{step_key} 使用命令: {label}")
        return label
    except Exception as e:
        reason = f"runtime failed: {pipeline._short_text(e)}"
        pipeline._last_aberration_api_error = reason
        pipeline.log.warn(f"{step_key} Aberration API {reason}")
        if pipeline.cfg.debug_mode:
            pipeline.log.debug(traceback.format_exc())
        return None

def load_sasp_stage8_module(pipeline):
    if pipeline._sasp_stage8_module is not None:
        return pipeline._sasp_stage8_module
    if pipeline._sasp_stage8_module_error is not None:
        return None

    wheel_path = pipeline._find_latest_sasp_wheel()
    if wheel_path is None:
        pipeline._sasp_stage8_module_error = (
            "setiastrosuitepro wheel not found in plugin downloads"
        )
        return None

    wheel_token = str(wheel_path)
    if wheel_token not in sys.path:
        sys.path.insert(0, wheel_token)

    pipeline._install_pyqt6_headless_stub()
    try:
        pipeline._install_sasp_stage8_widget_import_shims(wheel_path)
    except Exception as e:
        pipeline.log.warn(f"SASP Starless 深加工 widget shim failed: {e}")
    try:
        module = importlib.import_module("setiastro.saspro.wavescalede")
    except Exception as e:
        pipeline._sasp_stage8_module_error = f"import failed: {pipeline._short_text(e)}"
        pipeline.log.warn(f"SASP Starless 深加工 API import failed: {e}")
        if pipeline.cfg.debug_mode:
            pipeline.log.debug(traceback.format_exc())
        return None

    runner = getattr(module, "compute_wavescale_dse", None)
    if not callable(runner):
        pipeline._sasp_stage8_module_error = "compute_wavescale_dse is missing"
        pipeline.log.warn("SASP Starless 深加工 API unavailable: missing compute_wavescale_dse")
        return None

    pipeline._sasp_stage8_module = module
    pipeline._sasp_stage8_module_error = None
    return module

def install_sasp_stage8_widget_import_shims(pipeline, wheel_path: Path) -> None:
    """
    Load only the SASP widget pieces required by wavescalede.

    Importing setiastro.saspro.widgets normally executes widgets/__init__.py,
    which pulls preview/legacy GUI modules and can require cv2.  Stage 8 only
    needs wavelet_utils plus two UI symbols for class definitions, so keep
    the import surface narrow and avoid making OpenCV a hard dependency.
    """
    widgets_pkg_name = "setiastro.saspro.widgets"
    widgets_pkg = types.ModuleType(widgets_pkg_name)
    widgets_pkg.__path__ = [f"{wheel_path}/setiastro/saspro/widgets"]  # type: ignore[attr-defined]
    sys.modules[widgets_pkg_name] = widgets_pkg

    themed_name = f"{widgets_pkg_name}.themed_buttons"
    themed_mod = types.ModuleType(themed_name)

    class _Stage8DummyWidget:
        def __init__(self, *_args, **_kwargs):
            pass

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    def _themed_toolbtn(*_args, **_kwargs):
        return _Stage8DummyWidget()

    themed_mod.themed_toolbtn = _themed_toolbtn  # type: ignore[attr-defined]
    sys.modules[themed_name] = themed_mod
    setattr(widgets_pkg, "themed_buttons", themed_mod)

    graphics_name = f"{widgets_pkg_name}.graphics_views"
    graphics_mod = types.ModuleType(graphics_name)
    graphics_mod.ZoomableGraphicsView = type(  # type: ignore[attr-defined]
        "ZoomableGraphicsView",
        (_Stage8DummyWidget,),
        {},
    )
    sys.modules[graphics_name] = graphics_mod
    setattr(widgets_pkg, "graphics_views", graphics_mod)

    wavelet_name = f"{widgets_pkg_name}.wavelet_utils"
    if wavelet_name not in sys.modules:
        member = "setiastro/saspro/widgets/wavelet_utils.py"
        with zipfile.ZipFile(wheel_path) as zf:
            source = zf.read(member).decode("utf-8")
        wavelet_mod = types.ModuleType(wavelet_name)
        wavelet_mod.__file__ = f"{wheel_path}/{member}"
        sys.modules[wavelet_name] = wavelet_mod
        exec(compile(source, wavelet_mod.__file__, "exec"), wavelet_mod.__dict__)
        setattr(widgets_pkg, "wavelet_utils", wavelet_mod)

def prepare_stage8_sasp_input(pipeline, image_data):
    arr = np.asarray(image_data)
    dtype = arr.dtype
    scale_back: Optional[float] = None
    if np.issubdtype(dtype, np.integer):
        max_value = float(np.iinfo(dtype).max)
        working = arr.astype(np.float32) / max_value
        scale_back = max_value
    else:
        working = arr.astype(np.float32, copy=False)

    if working.ndim == 2:
        return np.clip(working, 0.0, 1.0), "mono", dtype, scale_back
    if working.ndim != 3:
        raise ValueError(f"unsupported image shape: {working.shape}")

    if working.shape[0] in (1, 3):
        hwc = np.transpose(working, (1, 2, 0))
        return np.clip(hwc, 0.0, 1.0), "chw", dtype, scale_back
    if working.shape[2] in (1, 3):
        return np.clip(working, 0.0, 1.0), "hwc", dtype, scale_back
    raise ValueError(f"unsupported channel layout: {working.shape}")

def restore_stage8_sasp_output(
    pipeline,
    output_data,
    layout: str,
    src_dtype,
    scale_back: Optional[float],
):
    out = np.asarray(output_data, dtype=np.float32)

    if layout == "mono":
        if out.ndim == 3:
            if out.shape[2] not in (1, 3):
                raise ValueError(f"unexpected SASP stage8 output shape: {out.shape}")
            out = out[:, :, 0]
        elif out.ndim != 2:
            raise ValueError(f"unexpected SASP stage8 output shape: {out.shape}")
    elif layout == "chw":
        if out.ndim == 2:
            out = out[:, :, None]
        if out.ndim != 3 or out.shape[2] not in (1, 3):
            raise ValueError(f"unexpected SASP stage8 output shape: {out.shape}")
        out = np.transpose(out, (2, 0, 1))
    elif layout == "hwc":
        if out.ndim == 2:
            out = out[:, :, None]
        if out.ndim != 3 or out.shape[2] not in (1, 3):
            raise ValueError(f"unexpected SASP stage8 output shape: {out.shape}")
    else:
        raise ValueError(f"unknown layout: {layout}")

    out = np.clip(out, 0.0, 1.0)
    if np.issubdtype(src_dtype, np.integer):
        max_value = scale_back if scale_back is not None else float(np.iinfo(src_dtype).max)
        return np.clip(out * max_value, 0, max_value).astype(src_dtype, copy=False)
    return out.astype(np.float32, copy=False)

def run_sasp_stage8_api(pipeline, plan: Optional[Dict[str, Any]] = None):
    pipeline._last_sasp_stage8_error = None
    module = pipeline._load_sasp_stage8_module()
    if module is None:
        reason = pipeline._sasp_stage8_module_error or "module unavailable"
        pipeline._last_sasp_stage8_error = reason
        pipeline.log.warn(f"SASP Starless 深加工 API unavailable: {reason}")
        return None

    def _run_with_pixels() -> np.ndarray:
        image_data = pipeline.siril.get_image_pixeldata(preview=False)
        if image_data is None:
            raise RuntimeError("image buffer is empty")
        source_data = np.asarray(image_data)
        input_image, input_layout, input_dtype, scale_back = pipeline._prepare_stage8_sasp_input(
            source_data
        )
        saturation = float((plan or {}).get("saturation", pipeline.cfg.nebula_saturation))
        source_features = measure_image_features(source_data)
        source_quality = measure_quality_metrics(source_data)
        boost_upper = 1.32
        gamma_upper = 1.18
        if (
            source_features.bg_std > pipeline.cfg.stage7_bg_std_high
            or source_quality.highlight_clip_ratio > pipeline.cfg.stage8_highlight_clip_ratio_max
        ):
            boost_upper = 1.18
            gamma_upper = 1.08
        boost = _clamp_float(1.05 + saturation * 0.45, 1.08, boost_upper)
        gamma = _clamp_float(1.0 + saturation * 0.25, 1.0, gamma_upper)
        output_image, _mask = module.compute_wavescale_dse(
            input_image,
            n_scales=6,
            boost_factor=boost,
            mask_gamma=gamma,
            iterations=1,
            decay_rate=0.45,
        )
        restored = pipeline._restore_stage8_sasp_output(
            output_image,
            input_layout,
            input_dtype,
            scale_back,
        )
        if pipeline.cfg.stage8_masked_enhancement_enabled:
            blended, diagnostics, messages = pipeline._apply_stage8_masked_pixel_enhancement(
                source_data,
                plan or {},
                label="SASP",
                plugin_candidate=restored,
            )
            diagnostics["sasp_params"] = {
                "boost_factor": boost,
                "mask_gamma": gamma,
                "boost_upper": boost_upper,
                "mask_gamma_upper": gamma_upper,
            }
            pipeline._last_stage8_masked_diagnostics = diagnostics
            for message in messages:
                pipeline.log.info(message)
            pipeline.siril.set_image_pixeldata(blended)
            return blended
        pipeline.siril.set_image_pixeldata(restored)
        return restored

    try:
        lock_factory = getattr(pipeline.siril, "image_lock", None)
        if callable(lock_factory):
            with lock_factory():
                _run_with_pixels()
        else:
            pipeline.log.warn(
                "SASP Starless 深加工 API: image_lock unavailable, running without thread lock"
            )
            _run_with_pixels()

        label = "SASP WaveScale Dark Enhancer API"
        pipeline.workflow_command_used["SASP Starless 深加工 API"] = label
        pipeline.log.info(f"SASP Starless 深加工 API 使用命令: {label}")
        return label
    except Exception as e:
        reason = f"runtime failed: {pipeline._short_text(e)}"
        pipeline._last_sasp_stage8_error = reason
        pipeline.log.warn(f"SASP Starless 深加工 API {reason}")
        if pipeline.cfg.debug_mode:
            pipeline.log.debug(traceback.format_exc())
        return None
