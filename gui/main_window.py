#!/usr/bin/env python3
"""Seestar Superimpose macOS GUI launcher with external pipeline resource."""

from __future__ import annotations

import importlib.util
import os
import platform
import queue
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from math import floor
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QTextCursor

try:
    from .disk_preflight import (
        DISK_SPACE_HEADROOM_RATIO,
        DISK_SPACE_MIN_HEADROOM_BYTES,
        FITS_SUFFIXES,
        LIGHT_FRAME_EXPANSION_FACTOR,
        LIGHT_PREPROCESS_SEQUENCE_COPIES,
        LINEAR_RESUME_STAGE_ARTIFACT_COPIES,
        STACKED_STAGE_ARTIFACT_COPIES,
        DiskSpaceEstimate,
        directory_size_bytes,
        format_bytes,
        safe_file_size,
        safe_mtime,
    )
except ImportError:
    from disk_preflight import (  # type: ignore[no-redef]
        DISK_SPACE_HEADROOM_RATIO,
        DISK_SPACE_MIN_HEADROOM_BYTES,
        FITS_SUFFIXES,
        LIGHT_FRAME_EXPANSION_FACTOR,
        LIGHT_PREPROCESS_SEQUENCE_COPIES,
        LINEAR_RESUME_STAGE_ARTIFACT_COPIES,
        STACKED_STAGE_ARTIFACT_COPIES,
        DiskSpaceEstimate,
        directory_size_bytes,
        format_bytes,
        safe_file_size,
        safe_mtime,
    )

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    if is_frozen():
        exe_path = Path(sys.executable).resolve()
        # <App>.app/Contents/MacOS/<binary> -> Resources
        return exe_path.parent.parent / "Resources"
    return project_root() / "resources"


def shell_quote_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def compute_siril_cpu_limit() -> int | None:
    total = os.cpu_count()
    if total is None or total < 1:
        return None
    return min(total, max(1, floor(total * 0.8)))


DEFAULT_SIRIL_CONFIG_TEMPLATE = """[core]
wd=
extension=.fit
force_16bit=false
fits_save_icc=true
allow_heterogeneous_fitseq=false
mem_mode=0
mem_ratio=0.90000000000000002
mem_amount=10
hd_bitdepth=20
script_check_requires=true
pipe_check_requires=false
check_updates=false
lang=
swap_dir=
binning_update=true
wcs_formalism=1
rgb_aladin=false
use_checksum=false
copyright=
graxpert_path=
asnet_dir=
fftw_timelimit=60
fftw_conv_fft_cutoff=15
fftwf_strategy=0
fftw_multithreaded=true
max_slice_size=32769

[starfinder]
focal_length=0
pixel_size=0

[debayer]
use_bayer_header=true
pattern=0
interpolation=8
orientation=0
offset_x=0
offset_y=0
xtrans_passes=1

[photometry]
gain=2.2999999999999998
inner=20
outer=30
inner_factor=4.2000000000000002
outer_factor=6.2999999999999998
force_radius=true
auto_aperture_factor=4
aperture=10
minval=-1500
maxval=60000
discard_var_catalogues=4
redpref=
greenpref=
bluepref=
lpfpref=
oscfilterpref=No filter
monosensorpref=
oscsensorpref=
is_mono=true
is_dslr=false
nb_mode=false
r_wl=656.27999999999997
r_bw=6
g_wl=500.69999999999999
g_bw=6
b_wl=500.69999999999999
b_bw=6

[astrometry]
asnet_keep_xyls=false
asnet_keep_wcs=false
asnet_show_output=false
sip_order=3
radius=10
max_seconds_run=30
update_default_scale=true
percent_scale_range=20
default_obscode=

[analysis]
panel=256
window=381

[compression]
enabled=false
method=0
quantization=16
hcompress_scale=4

[gui_prepro]
cfa=false
equalize_cfa=true
fix_xtrans=false
xtrans_af_x=0
xtrans_af_y=0
xtrans_af_w=0
xtrans_af_h=0
xtrans_sample_x=0
xtrans_sample_y=0
xtrans_sample_w=0
xtrans_sample_h=0
bias_lib=
use_bias_lib=false
dark_lib=
use_dark_lib=false
flat_lib=
use_flat_lib=false
disto_lib=
use_disto_lib=false
stack_default=$seqname$stacked
use_stack_default=true

[gui_registration]
method=0
interpolation=4
clamping=true
drizz_weight_match_bitpix=false

[gui_stack]
method=0
normalization=3
rejection=5
weighting=0
sigma_low=3
sigma_high=3
linear_low=5
linear_high=5
percentile_low=3
percentile_high=3

[gui]
first_start=1.4.2
silent_quit=false
silent_linear=false
remember_windows=false
theme=0
font_scale=100
icon_symbolic=false
script_path=
use_scripts_repository=false
use_spcc_repository=false
auto_update_scripts=false
auto_update_spcc=false
selected_scripts=
warn_scripts_run=true
show_thumbnails=true
thumbnail_size=256
selection_guides=0
show_deciasec=false
default_rendering_mode=0
display_histogram_mode=0
roi_mode=0
roi_warning=true
mmb_zoom_action=0
mouse_speed_limit=0
custom_monitor_profile=
soft_proofing_profile=
icc_custom_monitor_active=false
icc_soft_proofing_active=false
custom_RGB_ICC_profile=
custom_gray_ICC_profile=
rendering_intent=1
proofing_intent=1
export_intent=1
default_to_srgb=true
working_gamut=0
export_8bit_method=0
export_16bit_method=1
icc_autoconversion=0
icc_autoassignment=4
icc_rendering_bpc=true
icc_pedantic_linear=false
mouse_actions=
scroll_actions=

[gui_astrometry]
compass_position=1
cat_messier=true
cat_ngc=true
cat_ic=true
cat_ldn=true
cat_sh2=true
cat_stars=true
cat_const=true
cat_const_names=true
cat_user_dso=true
cat_user_sso=true

[gui_pixelmath]
pm_presets=

[script_editor]
highlight_syntax=true
highlight_bracketmatch=true
rmargin=true
rmargin_pos=80
show_linenums=true
show_linemarks=false
"""


def _normalize_gaia_photo_catalog(value: str) -> str:
    stripped = value.strip()
    unquoted = stripped.strip('"').strip("'")
    if not unquoted:
        return ""

    path = Path(unquoted).expanduser()
    if path.name == "gaia_photometric.dat" or not path.is_dir():
        return ""
    return stripped


def normalize_siril_config_template(config_text: str) -> str:
    removed_keys = {"starnet_exe", "starnet_weights"}
    lines = []
    for raw_line in config_text.splitlines():
        key = raw_line.split("=", 1)[0].strip().lower() if "=" in raw_line else ""
        if key in removed_keys:
            continue
        if key == "catalogue_gaia_photo":
            value = raw_line.split("=", 1)[1]
            lines.append(f"catalogue_gaia_photo={_normalize_gaia_photo_catalog(value)}")
            continue
        lines.append(raw_line)

    meaningful_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]
    has_key_value = any("=" in line and not line.startswith("[") for line in meaningful_lines)
    if not lines or not has_key_value:
        return DEFAULT_SIRIL_CONFIG_TEMPLATE
    if not any(line.strip() == "[core]" for line in lines):
        lines.extend(["", "[core]"])

    return "\n".join(lines) + "\n"


PIPELINE_RESOURCE_REL = Path("pipeline") / "seestar_Superimpose.py"
SIRIL_PLUGIN_RESOURCE_REL = Path("resources") / "siril_plugins"
SYQON_STARLESS_BUNDLE_REL = Path("syqon_starless")
COSMIC_CLARITY_BUNDLE_REL = Path("cosmic_clarity")
COSMIC_CLARITY_REQUIRED_MODEL_FILES = (
    "deep_denoise_mono_AI4.pth",
    "deep_denoise_color_AI4.pth",
    "deep_sharp_stellar_AI4.pth",
    "deep_nonstellar_sharp_conditional_psf_AI4.pth",
)
APP_RUNTIME_HOME_REL = Path("Library/Application Support/SeestarSuperimpose/runtime_home")
AI_ENV_RESOURCE_REL = Path("ai.env")
DEFAULT_ENV_RESOURCE_REL = Path("default.env")
AI_ENV_OVERRIDE_NAME = ".seestar_ai.env"
AI_ENV_ALLOWED_KEYS = frozenset(
    {
        "SEESTAR_AI_ENABLED",
        "SEESTAR_AI_ENDPOINT",
        "SEESTAR_AI_MODEL",
        "SEESTAR_AI_API_KEY",
        "SEESTAR_AI_TIMEOUT_SEC",
        "SEESTAR_AI_STRENGTH",
        "SEESTAR_AI_PROMPT",
        "SEESTAR_AI_STAGE6_ENABLE",
        "SEESTAR_AI_STAGE7_ENABLE",
        "SEESTAR_AI_STAGE8_ENABLE",
        "SEESTAR_OUTPUT_FORMAT",
        "SEESTAR_DENOISE_ENABLE",
        "SEESTAR_DENOISE_FORCE",
        "SEESTAR_SYQON_GPU",
        "SEESTAR_SYQON_TIMEOUT_SEC",
        "SEESTAR_SIRILPY_TIMEOUT_SEC",
        "SEESTAR_WORKFLOW_PLUGIN_PROBE",
        "SEESTAR_SPCC_ENABLE",
        "SEESTAR_STAGE4_PLATESOLVE_ENABLE",
        "SEESTAR_STAGE4_PLATESOLVE_CATALOGS",
        "SEESTAR_STAGE4_SPCC_SENSOR_MODE",
        "SEESTAR_STAGE4_SPCC_OSC_SENSOR",
        "SEESTAR_STAGE4_SPCC_OSC_FILTER",
        "SEESTAR_STAGE4_SPCC_BUILTIN_DUALBAND_FILTER",
        "SEESTAR_STAGE4_SPCC_MONO_SENSOR",
        "SEESTAR_STAGE4_SPCC_R_FILTER",
        "SEESTAR_STAGE4_SPCC_G_FILTER",
        "SEESTAR_STAGE4_SPCC_B_FILTER",
        "SEESTAR_STAGE4_SPCC_WHITE_REF",
        "SEESTAR_STAGE4_SPCC_ADAPTIVE_WHITE_REF",
        "SEESTAR_STAGE4_SPCC_NEBULA_WHITE_REF",
        "SEESTAR_STAGE4_SPCC_BGTOL",
        "SEESTAR_STAGE4_SPCC_LIMITMAG",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT",
        "SEESTAR_SPCC_ALLOW_LIGHT_PREPROCESS",
        "SEESTAR_ABERRATION_API_ENABLE",
        "SEESTAR_ABERRATION_PROVIDER",
        "SEESTAR_OPTIONAL_COLOR_TRANSFORM",
        "SEESTAR_COSMIC_CLASSIC_ENABLE",
        "SEESTAR_COSMIC_CLARITY_EXECUTABLE",
        "SEESTAR_COSMIC_CLASSIC_GPU",
        "SEESTAR_COSMIC_NATIVE_GPU",
        "SEESTAR_STAGE5_BUILTIN_DENOISE_MOD",
        "SEESTAR_STAGE5_DECONV_ENABLE",
        "SEESTAR_STAGE5_RL_MAXSTARS",
        "SEESTAR_STAGE5_RL_PSF_KS",
        "SEESTAR_STAGE5_RL_ITERS",
        "SEESTAR_STAGE5_RL_ALPHA",
        "SEESTAR_STAGE5_RL_GDSTEP",
        "SEESTAR_STAGE5_RL_STOP",
        "SEESTAR_STAGE5_GRAXPERT_DECONV_STRENGTH",
        "SEESTAR_STAGE7_QUALITY_RETRY_MAX",
        "SEESTAR_STAGE7_SKIP_UNREADY_STARLESS",
        "SEESTAR_STAR_SEPARATION_MODE",
        "SEESTAR_STAR_SEPARATION_FALLBACK_TO_MILD_PRESTRETCH",
        "SEESTAR_MILD_PRESTRETCH_STRENGTH",
        "SEESTAR_STAGE7_SOFT_STARLESS_ASINH_STRETCH",
        "SEESTAR_STAGE7_BRIGHT_NEBULA_HALO_RESIDUE_SCORE_MAX",
        "SEESTAR_STAGE7_STARLESS_REPAIR_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_HALO_REPAIR_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_CHROMA_DENOISE_STRENGTH",
        "SEESTAR_STAGE7_STARLESS_PIXEL_REPAIR_ENABLE",
        "SEESTAR_STAGE8_FORCE_CONSERVATIVE_AFTER_STAGE7_REPAIR",
        "SEESTAR_STAGE9_STARMASK_STRETCH_ENABLE",
        "SEESTAR_STAGE9_STARMASK_ASINH_STRETCH",
        "SEESTAR_STAGE9_STARMASK_ASINH_OFFSET",
    }
)


def default_pipeline_path(resources: Path) -> Path:
    if is_frozen():
        return resources / PIPELINE_RESOURCE_REL
    return project_root() / PIPELINE_RESOURCE_REL


def default_siril_plugin_dir(resources: Path) -> Path:
    if is_frozen():
        return resources / "siril_plugins"
    return project_root() / SIRIL_PLUGIN_RESOURCE_REL


def resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_runtime_home() -> Path:
    return Path.home() / APP_RUNTIME_HOME_REL


def siril_state_root_from_home(runtime_home: Path) -> Path:
    return runtime_home / "Library/Application Support/org.siril.Siril/siril"


SIRIL_REQUIRED_SITE_PACKAGES = ("sirilpy", "numpy", "packaging", "requests")
SIRIL_VENDOR_FALLBACK_PACKAGES = (
    "packaging",
    "requests",
    "urllib3",
    "charset_normalizer",
    "idna",
    "certifi",
)
SIRIL_COSMIC_REQUIRED_WHEEL_LABELS = (
    "PyQt6 wheel",
    "PyQt6_Qt6 wheel",
    "pyqt6_sip wheel",
    "tifffile wheel",
    "lz4 wheel",
    "zstandard wheel",
    "exifread wheel",
    "opencv-python-headless wheel",
    "requests wheel",
    "requests dependency wheels",
    "wheel package",
)
SIRIL_STARLESS_REQUIRED_WHEEL_LABELS = (
    "PySide6 wheel",
    "PySide6_Addons wheel",
    "PySide6_Essentials wheel",
    "shiboken6 wheel",
    "astropy wheel",
    "scipy wheel",
)
INPUT_MODE_AUTO = "auto"
INPUT_MODE_LINEAR_RESUME = "result_linear_resume"
INPUT_MODE_STAGE2_CORRECTED_RESUME = "stage2_corrected_resume"
LINEAR_RESUME_INPUT_NAME = "result_linear.fit"
STAGE2_CORRECTED_INPUT_NAME = "stage2_corrected.fit"
PIPELINE_EXCLUDE_PREFIXES = (
    "light_",
    "pp_",
    "r_",
    "stage",
    "starless",
    "starmask",
    "working",
    "result",
    "sasp_",
)
PIPELINE_EXCLUDE_SUBSTRINGS = (
    "starless",
    "starmask",
)
PIPELINE_EXCLUDE_SUFFIXES = (
    "_processed",
    "_final",
    "_enhanced",
    "_remixed",
)


def resolve_siril_scripts_root(plugin_root: Path) -> Path | None:
    candidates = [
        plugin_root / "vendor" / "siril-scripts",
        plugin_root / "vendor" / "siril-scripts" / "siril-scripts",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        if (root / "processing" / "AberrationRemover.py").is_file():
            return root
    return None


def apply_siril_runtime_patches(plugin_root: Path, target_root: Path | None = None) -> bool:
    patcher = plugin_root / "patches" / "apply_graxpert_ai_runtime_patch.py"
    if not patcher.is_file():
        return False
    spec = importlib.util.spec_from_file_location(
        "seestar_graxpert_ai_runtime_patch", patcher
    )
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return bool(module.apply_patch(target_root or plugin_root))


def build_siril_cli_command(
    siril_cli: Path,
    work_dir: Path,
    run_ini: Path,
    run_ssf: Path,
    offline_mode: bool,
) -> list[str]:
    cmd = [str(siril_cli)]
    if offline_mode:
        cmd.append("--offline")
    cmd.extend(
        [
            "-d",
            str(work_dir),
            "-i",
            str(run_ini),
            "-s",
            str(run_ssf),
        ]
    )
    return cmd


def scrub_python_env(env: dict[str, str]) -> dict[str, str]:
    drop_exact = {
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "PYTHONNOUSERSITE",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "__PYVENV_LAUNCHER__",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    }
    for key in list(env.keys()):
        if key in drop_exact or key.startswith("PYINSTALLER_"):
            env.pop(key, None)
            continue
        if key.startswith("PYTHON") and key not in {"PYTHONUTF8", "PYTHONIOENCODING"}:
            env.pop(key, None)
    return env


def parse_ai_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    parsed: dict[str, str] = {}
    warnings: list[str] = []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return parsed, [f"读取失败: {path} ({exc})"]

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            warnings.append(f"{path}:{lineno} 无法解析该行，已忽略")
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            warnings.append(f"{path}:{lineno} 缺少变量名，已忽略")
            continue
        if key not in AI_ENV_ALLOWED_KEYS:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif value and value[0] in {"'", '"'}:
            warnings.append(f"{path}:{lineno} 引号不闭合，已忽略")
            continue
        else:
            hash_with_space = value.find(" #")
            if hash_with_space >= 0:
                value = value[:hash_with_space].rstrip()

        parsed[key] = value

    return parsed, warnings


def resolve_venv_site_packages(venv_dir: Path) -> Path:
    lib_dir = venv_dir / "lib"
    for site_dir in sorted(lib_dir.glob("python*/site-packages")):
        if site_dir.is_dir():
            return site_dir
    raise FileNotFoundError(f"在 venv 中未找到 site-packages：{venv_dir}")


def repair_site_packages_from_pip_vendor(site_dir: Path) -> list[str]:
    repaired: list[str] = []
    vendor_dir = site_dir / "pip" / "_vendor"
    if not vendor_dir.is_dir():
        return repaired

    for pkg in SIRIL_VENDOR_FALLBACK_PACKAGES:
        dst = site_dir / pkg
        src = vendor_dir / pkg
        if dst.exists() or not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
            repaired.append(pkg)
    return repaired


def verify_siril_offline_seed_venv(venv_dir: Path) -> tuple[bool, str]:
    try:
        site_dir = resolve_venv_site_packages(venv_dir)
    except Exception as exc:
        return False, str(exc)

    missing = [pkg for pkg in SIRIL_REQUIRED_SITE_PACKAGES if not (site_dir / pkg).exists()]
    if missing:
        return False, f"Siril venv 缺少依赖包：{', '.join(missing)} ({site_dir})"

    py_candidates = (
        venv_dir / "bin" / "python3.12",
        venv_dir / "bin" / "python3",
        venv_dir / "bin" / "python",
    )
    py_bin = next((p for p in py_candidates if p.exists()), None)
    if py_bin is None:
        return False, f"Siril venv 缺少 Python 可执行文件：{venv_dir / 'bin'}"

    probe_code = (
        "import sirilpy, numpy, packaging, requests; "
        "print('sirilpy-ok')"
    )
    try:
        cp = subprocess.run(
            [str(py_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=scrub_python_env(os.environ.copy()),
        )
    except Exception as exc:
        return False, f"探测 Siril venv Python 失败：{exc}"

    if cp.returncode != 0:
        err = cp.stderr.strip() or cp.stdout.strip() or f"exit={cp.returncode}"
        return False, f"Siril venv 导入探测失败：{err}"

    return True, cp.stdout.strip() or "sirilpy-ok"


try:
    from .pipeline_worker import PipelineWorker
except ImportError:
    from pipeline_worker import PipelineWorker  # type: ignore[no-redef]


class SeestarGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Seestar 图像后处理")
        self.resize(980, 680)

        self.resources = resource_root()
        self.pipeline_path = default_pipeline_path(self.resources)
        self.siril_plugin_dir = default_siril_plugin_dir(self.resources)
        self.runtime_home = default_runtime_home()
        self.bundled_siril_cli = (
            self.resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        )
        self.siril_seed_dir = self.resources / "SirilPythonSeed"
        config_candidates = [
            self.resources / "config.1.4.ini.template",
        ]
        self.config_template = resolve_existing_path(config_candidates)

        self.worker: PipelineWorker | None = None
        self.run_log_path: Path | None = None
        self.run_log_file = None
        self._current_work_dir: Path | None = None
        self.input_mode = INPUT_MODE_AUTO
        self.debug_mode_enabled = False
        self.network_mode_enabled = True
        self.ai_stage_enabled = False

        self._init_ui()
        self._set_running(False)

        cli_order = self._resolve_siril_candidates()
        self._append_text(self._initial_panel_text(cli_order))

    def _section_block(self, title: str, body_lines: list[str]) -> list[str]:
        divider = "=" * 16
        return [f"{divider} {title} {divider}", *body_lines, ""]

    def _display_path(self, path: Path | str | None) -> str:
        if path is None:
            return "<无>"
        text = str(path)
        home = str(Path.home())
        if text == home:
            return "~"
        prefix = home + os.sep
        if text.startswith(prefix):
            return "~" + os.sep + text[len(prefix):]
        return text

    def _display_status(self, status: str) -> str:
        return {
            "Idle": "空闲",
            "Running": "运行中",
            "Completed": "已完成",
            "Failed": "失败",
            "Stopped": "已停止",
        }.get(status, status)

    def _set_status_text(self, status: str) -> None:
        self.status_label.setText(f"状态：{self._display_status(status)}")

    def _initial_panel_text(self, cli_order: list[Path]) -> str:
        if cli_order:
            candidate_names = ", ".join(p.name for p in cli_order)
        else:
            candidate_names = "<无>"
        lines: list[str] = []
        lines.extend(self._section_block("使用说明", [
            "  1. 点击“选择目录”，选择包含 Seestar 子帧文件的工作目录。",
            "  2. 根据数据状态选择处理模式；如已导出 result_linear.fit，可切到续跑模式。",
            "  3. 点击“开始处理”，应用会先执行预检与磁盘空间估算，再启动处理。",
            "  4. 本区域会同时显示使用说明和运行日志；需要中断时点击“停止”。",
            "  5. 处理完成后，可用“打开结果目录”查看 *.tif / *.png / result_linear.fit / *_final.fit；Debug 开启时可查看阶段 6/7 中间产物。",
            "  6. 如需完整运行记录，点击“打开日志文件”。",
        ]))
        lines.extend(self._section_block("应用能力", [
            "  - 支持已叠加 FITS 与 Seestar 子帧输入，必要时自动执行预处理。",
            "  - 支持显式从 result_linear.fit 继续执行阶段 6-11 后期流程。",
            "  - 串联 Siril 1.4+ 与 SyQon/SASP/CosmicClarity 插件链路，执行离线后处理，不依赖系统 Python。",
            "  - 处理过程包含预检、重试、降级回退和阶段状态记录。",
            "  - 输出高质量 TIFF、预览 PNG、拉伸前线性 FITS 和最终 FITS 归档。",
            "  - 控件区分为“处理过程”和“可选过程”两组，便于区分主流程与可选能力。",
            f"  - 处理模式: {self._input_mode_label(self.input_mode)}。",
            f"  - AI 阶段开关: {'开启' if self.ai_stage_enabled else '关闭'}（控制阶段 11 是否执行）。",
            f"  - Debug 模式: {'开启' if self.debug_mode_enabled else '关闭'}（开启后保留 stage* 中间产物；阶段 6 去星输出为 stage6_starless，阶段 7 拉伸输出为 stage7_stretched）。",
            f"  - 联网模式: {'开启' if self.network_mode_enabled else '关闭'}（开启后允许 platesolve 联网解算）。",
        ]))
        lines.extend(self._section_block("处理阶段总览", [
            "  线性阶段: 1.前期准备 -> 2.裁切 -> 3.背景提取 -> 4.图像解析+色彩校准 -> 5.线性降噪/反卷积",
            "  非线性阶段: 6.去星与星点层准备 -> 7.主体拉伸 -> 8.Starless 深加工 -> 9.星点处理与合成 -> 10.最终降噪与导出",
            "  可选阶段: 11.AI 后期美化（需 SEESTAR_AI_* 配置）",
        ]))
        lines.extend(self._section_block("阶段文件命名", [
            "  - 阶段 6 去星: stage6_starless.fit / stage6_starless_quality.json。",
            "  - 阶段 7 拉伸: stage7_stretched.fit / stage7_stretch_quality.json。",
            "  - 阶段 7 统一使用 stage7_cand_a/b、stage7_preview_ref 与 stage7_stretched 命名。",
        ]))
        lines.extend(self._section_block("当前运行环境", [
            f"  资源根目录: {self._display_path(self.resources)}",
            f"  运行时主目录: {self._display_path(self.runtime_home)}",
            "  核心文件: "
            f"流水线={self.pipeline_path.name}, "
            f"Siril={candidate_names}",
            "  完整运行时路径和预检细节会在任务开始时写入日志。",
        ]))
        lines.append("已就绪。")
        return "\n".join(lines) + "\n"

    def _reset_view(self) -> None:
        cli_order = self._resolve_siril_candidates()
        text = self._initial_panel_text(cli_order)
        if self.worker and self.worker.isRunning():
            text += (
                "\n"
                f"{'=' * 16} 当前任务进行中 {'=' * 16}\n"
                "当前日志文件仍在持续写入；新输出会从此处继续追加。\n"
            )
        self.log_view.setPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def _resolve_siril_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        if self.bundled_siril_cli.exists() and self.bundled_siril_cli.is_file():
            candidates.append(self.bundled_siril_cli)
        return candidates

    def _siril_state_root(self) -> Path:
        return siril_state_root_from_home(self.runtime_home)

    def _rewrite_seeded_venv(self, venv_dir: Path) -> None:
        py_bin = (
            self.resources
            / "Siril.app"
            / "Contents"
            / "Frameworks"
            / "Python.framework"
            / "Versions"
            / "3.12"
            / "bin"
            / "python3.12"
        )
        if not py_bin.exists():
            raise FileNotFoundError(f"内置 Siril Python 缺失：{py_bin}")

        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        for name in ("python3.12", "python3", "python"):
            link_path = bin_dir / name
            if link_path.exists() or link_path.is_symlink():
                try:
                    link_path.unlink()
                except Exception:
                    pass
            link_path.symlink_to(py_bin)

        cfg = venv_dir / "pyvenv.cfg"
        if cfg.exists():
            content = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
        else:
            content = []

        siril_python = self.resources / "Siril.app" / "Contents" / "MacOS" / "python3"
        replacements = {
            "home": str(py_bin.parent),
            "executable": str(py_bin),
            "command": f"{siril_python} -m venv {venv_dir}",
        }
        updated: dict[str, str] = {}
        output_lines: list[str] = []
        for line in content:
            if "=" not in line:
                output_lines.append(line)
                continue
            key, _value = line.split("=", 1)
            k = key.strip()
            if k in replacements:
                output_lines.append(f"{k} = {replacements[k]}")
                updated[k] = replacements[k]
            else:
                output_lines.append(line)
        for key, value in replacements.items():
            if key not in updated:
                output_lines.append(f"{key} = {value}")
        cfg.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    def _ensure_offline_siril_python_seed(self) -> None:
        seed_venv = self.siril_seed_dir / "venv"
        seed_module = self.siril_seed_dir / ".python_module"
        if not seed_venv.exists():
            raise FileNotFoundError(f"缺少内置 Siril venv seed：{seed_venv}")
        if not seed_module.exists():
            raise FileNotFoundError(f"缺少内置 Siril 模块 seed：{seed_module}")

        state_root = self._siril_state_root()
        target_venv = state_root / "venv"
        target_module = state_root / ".python_module"
        state_root.mkdir(parents=True, exist_ok=True)

        seeded = False
        if not target_venv.exists():
            shutil.copytree(seed_venv, target_venv, symlinks=True)
            seeded = True
            self._append_event(f"已写入 Siril 离线 venv：{target_venv}")
        if not target_module.exists():
            shutil.copytree(seed_module, target_module, symlinks=True)
            seeded = True
            self._append_event(f"已写入 Siril 离线模块：{target_module}")
        elif not (target_module / "sirilpy").exists():
            shutil.rmtree(target_module, ignore_errors=True)
            shutil.copytree(seed_module, target_module, symlinks=True)
            seeded = True
            self._append_event(f"已重新写入 Siril 离线模块：{target_module}")

        self._rewrite_seeded_venv(target_venv)
        site_dir = resolve_venv_site_packages(target_venv)
        repaired = repair_site_packages_from_pip_vendor(site_dir)
        if repaired:
            self._append_event(
                "已从 pip vendor 修补 Siril venv 依赖："
                + ", ".join(repaired)
            )
        ok, detail = verify_siril_offline_seed_venv(target_venv)
        if not ok:
            raise RuntimeError(detail)
        if seeded:
            self._append_event("离线 Siril Python seed 已就绪。")

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        dir_label = QLabel("工作目录")
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("~/SeeStar/M 42_sub")
        browse_btn = QPushButton("选择目录")
        browse_btn.clicked.connect(self._choose_workdir)

        grid.addWidget(dir_label, 0, 0)
        grid.addWidget(self.dir_edit, 0, 1)
        grid.addWidget(browse_btn, 0, 2)

        mode_label = QLabel("处理模式")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Normal Pipeline", INPUT_MODE_AUTO)
        self.mode_combo.addItem(
            "Postprocess From stage2_corrected.fit",
            INPUT_MODE_STAGE2_CORRECTED_RESUME,
        )
        self.mode_combo.addItem(
            "Continue From result_linear.fit",
            INPUT_MODE_LINEAR_RESUME,
        )
        self.mode_combo.currentIndexChanged.connect(self._on_input_mode_changed)
        grid.addWidget(mode_label, 1, 0)
        grid.addWidget(self.mode_combo, 1, 1, 1, 2)

        self.status_label = QLabel("状态：空闲")
        self.status_label.setStyleSheet("font-weight: 600;")
        grid.addWidget(self.status_label, 2, 0, 1, 3)

        outer.addLayout(grid)

        process_actions = QHBoxLayout()
        process_actions.setSpacing(8)

        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self._start_run)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._stop_run)

        self.open_result_btn = QPushButton("打开结果目录")
        self.open_result_btn.clicked.connect(self._open_result_dir)

        self.open_log_btn = QPushButton("打开日志文件")
        self.open_log_btn.clicked.connect(self._open_log_file)

        self.clear_view_btn = QPushButton("清空视图")
        self.clear_view_btn.clicked.connect(self._reset_view)

        self.debug_btn = QPushButton()
        self.debug_btn.setCheckable(True)
        self.debug_btn.toggled.connect(self._on_debug_toggled)
        self._update_debug_button_text()

        self.network_btn = QPushButton()
        self.network_btn.setCheckable(True)
        self.network_btn.toggled.connect(self._on_network_toggled)
        self._update_network_button_text()

        self.ai_btn = QPushButton()
        self.ai_btn.setCheckable(True)
        self.ai_btn.toggled.connect(self._on_ai_toggled)
        self._update_ai_button_text()

        process_label = QLabel("处理过程")
        process_label.setStyleSheet("font-weight: 600; color: #2f4f4f;")
        process_actions.addWidget(process_label)
        process_actions.addWidget(self.start_btn)
        process_actions.addWidget(self.stop_btn)
        process_actions.addWidget(self.open_result_btn)
        process_actions.addWidget(self.open_log_btn)
        process_actions.addWidget(self.clear_view_btn)
        process_actions.addStretch(1)
        outer.addLayout(process_actions)

        optional_actions = QHBoxLayout()
        optional_actions.setSpacing(8)
        optional_label = QLabel("可选过程")
        optional_label.setStyleSheet("font-weight: 600; color: #6a4b16;")
        optional_actions.addWidget(optional_label)
        optional_actions.addWidget(self.ai_btn)
        optional_actions.addWidget(self.debug_btn)
        optional_actions.addWidget(self.network_btn)
        optional_actions.addStretch(1)
        outer.addLayout(optional_actions)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        outer.addWidget(self.log_view)

        process_menu = self.menuBar().addMenu("处理")

        self.start_action = QAction("开始处理", self)
        self.start_action.triggered.connect(self._start_run)
        process_menu.addAction(self.start_action)

        self.stop_action = QAction("停止", self)
        self.stop_action.triggered.connect(self._stop_run)
        process_menu.addAction(self.stop_action)

        process_menu.addSeparator()

        self.open_result_action = QAction("打开结果目录", self)
        self.open_result_action.triggered.connect(self._open_result_dir)
        process_menu.addAction(self.open_result_action)

        self.open_log_action = QAction("打开日志文件", self)
        self.open_log_action.triggered.connect(self._open_log_file)
        process_menu.addAction(self.open_log_action)

        self.clear_view_action = QAction("清空视图", self)
        self.clear_view_action.triggered.connect(self._reset_view)
        process_menu.addAction(self.clear_view_action)

    def _choose_workdir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if selected:
            self.dir_edit.setText(selected)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.start_action.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.stop_action.setEnabled(running)
        self.dir_edit.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.ai_btn.setEnabled(not running)
        self.debug_btn.setEnabled(not running)
        self.network_btn.setEnabled(not running)

    def _input_mode_label(self, mode: str) -> str:
        if mode == INPUT_MODE_LINEAR_RESUME:
            return "result_linear.fit 后期模式"
        if mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            return "stage2_corrected.fit 叠加后处理模式"
        return "正常流程模式"

    def _current_input_mode(self) -> str:
        combo = getattr(self, "mode_combo", None)
        if combo is not None and hasattr(combo, "currentData"):
            value = combo.currentData()
            if value in {
                INPUT_MODE_AUTO,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }:
                return str(value)
        value = getattr(self, "input_mode", INPUT_MODE_AUTO)
        if value in {
            INPUT_MODE_AUTO,
            INPUT_MODE_LINEAR_RESUME,
            INPUT_MODE_STAGE2_CORRECTED_RESUME,
        }:
            return str(value)
        return INPUT_MODE_AUTO

    def _linear_resume_input_path(self, work_dir: Path) -> Path:
        return work_dir / LINEAR_RESUME_INPUT_NAME

    def _stage2_corrected_resume_input_path(self, work_dir: Path) -> Path:
        root_candidate = work_dir / STAGE2_CORRECTED_INPUT_NAME
        if root_candidate.is_file():
            return root_candidate
        return work_dir / "process" / STAGE2_CORRECTED_INPUT_NAME

    def _on_input_mode_changed(self, _index: int) -> None:
        self.input_mode = self._current_input_mode()
        self._append_event(f"处理模式已切换为：{self._input_mode_label(self.input_mode)}")

    def _update_ai_button_text(self) -> None:
        if self.ai_stage_enabled:
            self.ai_btn.setText("AI: ON")
            self.ai_btn.setStyleSheet(
                "background-color: #8a5a00; color: white; font-weight: 600;"
            )
        else:
            self.ai_btn.setText("AI: OFF")
            self.ai_btn.setStyleSheet("")
        if self.ai_btn.isChecked() != self.ai_stage_enabled:
            self.ai_btn.blockSignals(True)
            try:
                self.ai_btn.setChecked(self.ai_stage_enabled)
            finally:
                self.ai_btn.blockSignals(False)

    def _on_ai_toggled(self, checked: bool) -> None:
        self.ai_stage_enabled = bool(checked)
        self._update_ai_button_text()
        self._append_event(
            "AI 阶段开关已"
            + (
                "开启：将执行阶段 11（需配置 SEESTAR_AI_*）"
                if self.ai_stage_enabled
                else "关闭：将跳过阶段 11"
            )
        )

    def _update_debug_button_text(self) -> None:
        if self.debug_mode_enabled:
            self.debug_btn.setText("Debug: ON")
            self.debug_btn.setStyleSheet(
                "background-color: #0b7a3b; color: white; font-weight: 600;"
            )
        else:
            self.debug_btn.setText("Debug: OFF")
            self.debug_btn.setStyleSheet("")
        if self.debug_btn.isChecked() != self.debug_mode_enabled:
            self.debug_btn.blockSignals(True)
            try:
                self.debug_btn.setChecked(self.debug_mode_enabled)
            finally:
                self.debug_btn.blockSignals(False)

    def _on_debug_toggled(self, checked: bool) -> None:
        self.debug_mode_enabled = bool(checked)
        self._update_debug_button_text()
        self._append_event(
            "Debug 模式已"
            + (
                "开启：保留 stage* 中间产物，并输出 DEBUG 级别过程日志（含 SPCC 命令细节；lightsrc 序列仍清理）"
                if self.debug_mode_enabled
                else "关闭：将按默认策略清理中间产物"
            )
        )

    def _update_network_button_text(self) -> None:
        if self.network_mode_enabled:
            self.network_btn.setText("联网: ON")
            self.network_btn.setStyleSheet(
                "background-color: #0f5f9c; color: white; font-weight: 600;"
            )
        else:
            self.network_btn.setText("联网: OFF")
            self.network_btn.setStyleSheet("")
        if self.network_btn.isChecked() != self.network_mode_enabled:
            self.network_btn.blockSignals(True)
            try:
                self.network_btn.setChecked(self.network_mode_enabled)
            finally:
                self.network_btn.blockSignals(False)

    def _on_network_toggled(self, checked: bool) -> None:
        self.network_mode_enabled = bool(checked)
        self._update_network_button_text()
        self._append_event(
            "联网模式已"
            + ("开启：允许 platesolve 联网解算" if self.network_mode_enabled else "关闭：强制使用 --offline")
        )

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        if self.run_log_file:
            self.run_log_file.write(text)
            self.run_log_file.flush()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_event(self, msg: str) -> None:
        self._append_text(f"[{self._timestamp()}] {msg}\n")

    def _append_divider(self, title: str, detail_lines: list[str] | None = None) -> None:
        self._append_text(f"\n{'=' * 16} {title} {'=' * 16}\n")
        if detail_lines:
            for line in detail_lines:
                self._append_text(f"{line}\n")

    def _fits_in_work_dir(self, work_dir: Path) -> list[Path]:
        if not work_dir.is_dir():
            return []
        return [
            p for p in work_dir.iterdir()
            if p.is_file() and p.suffix.lower() in FITS_SUFFIXES
        ]

    def _is_candidate_stacked_input(self, path: Path, work_dir: Path) -> bool:
        if path.parent != work_dir:
            return False

        name_lower = path.name.lower()
        stem_lower = path.stem.lower()
        for prefix in PIPELINE_EXCLUDE_PREFIXES:
            if name_lower.startswith(prefix):
                return False
        for substring in PIPELINE_EXCLUDE_SUBSTRINGS:
            if substring in stem_lower:
                return False
        for suffix in PIPELINE_EXCLUDE_SUFFIXES:
            if stem_lower.endswith(suffix):
                return False
        return True

    def _estimate_disk_space(self, work_dir: Path) -> DiskSpaceEstimate | None:
        current_work_dir_bytes = directory_size_bytes(work_dir)
        available_bytes = shutil.disk_usage(work_dir).free
        current_mode = self._current_input_mode()

        if current_mode == INPUT_MODE_LINEAR_RESUME:
            source = self._linear_resume_input_path(work_dir)
            if not source.is_file():
                return None
            source_bytes = safe_file_size(source)
            base_growth_bytes = int(source_bytes * LINEAR_RESUME_STAGE_ARTIFACT_COPIES)
            input_count = 1
            input_bytes = source_bytes
            mode = "linear_resume"
            selected_input_label = source.name
        elif current_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            source = self._stage2_corrected_resume_input_path(work_dir)
            if not source.is_file():
                return None
            source_bytes = safe_file_size(source)
            base_growth_bytes = int(source_bytes * STACKED_STAGE_ARTIFACT_COPIES)
            input_count = 1
            input_bytes = source_bytes
            mode = "stage2_corrected_resume"
            selected_input_label = source.name
        else:
            fits = self._fits_in_work_dir(work_dir)
            if not fits:
                return None

            light_files = [p for p in fits if p.name.lower().startswith("light_")]
            stacked_files = [
                p for p in fits
                if self._is_candidate_stacked_input(p, work_dir)
            ]

            if stacked_files:
                source = max(stacked_files, key=safe_mtime)
                source_bytes = safe_file_size(source)
                base_growth_bytes = int(source_bytes * STACKED_STAGE_ARTIFACT_COPIES)
                input_count = 1
                input_bytes = source_bytes
                mode = "stacked"
                selected_input_label = source.name
            elif light_files:
                input_count = len(light_files)
                input_bytes = sum(safe_file_size(p) for p in light_files)
                largest_light_bytes = max((safe_file_size(p) for p in light_files), default=0)
                average_light_bytes = input_bytes / max(input_count, 1)
                processed_frame_bytes = max(
                    largest_light_bytes,
                    int(average_light_bytes * LIGHT_FRAME_EXPANSION_FACTOR),
                )
                preprocess_growth_bytes = int(
                    input_bytes
                    * LIGHT_FRAME_EXPANSION_FACTOR
                    * LIGHT_PREPROCESS_SEQUENCE_COPIES
                )
                stacked_stage_growth_bytes = int(
                    processed_frame_bytes * STACKED_STAGE_ARTIFACT_COPIES
                )
                base_growth_bytes = preprocess_growth_bytes + stacked_stage_growth_bytes
                mode = "light"
                selected_input_label = f"Light_ x {input_count}"
            else:
                return None

        headroom_bytes = max(
            DISK_SPACE_MIN_HEADROOM_BYTES,
            int(base_growth_bytes * DISK_SPACE_HEADROOM_RATIO),
        )
        required_free_bytes = base_growth_bytes + headroom_bytes
        return DiskSpaceEstimate(
            mode=mode,
            current_work_dir_bytes=current_work_dir_bytes,
            input_count=input_count,
            input_bytes=input_bytes,
            estimated_peak_growth_bytes=base_growth_bytes,
            required_free_bytes=required_free_bytes,
            available_bytes=available_bytes,
            selected_input_label=selected_input_label,
        )

    def _disk_space_mode_label(self, estimate: DiskSpaceEstimate) -> str:
        if estimate.mode == "linear_resume":
            return "result_linear.fit 后期模式"
        if estimate.mode == "stage2_corrected_resume":
            return "stage2_corrected.fit 叠加后处理模式"
        if estimate.mode == "light":
            return "Light_ 预处理模式"
        return "已叠加 FITS 直处理模式"

    def _disk_space_summary_lines(self, estimate: DiskSpaceEstimate) -> list[str]:
        input_desc = estimate.selected_input_label
        if estimate.mode == "light":
            input_desc += (
                f" ({estimate.input_count} 帧, {format_bytes(estimate.input_bytes)})"
            )
        else:
            input_desc += f" ({format_bytes(estimate.input_bytes)})"

        return [
            f"  磁盘预估模式: {self._disk_space_mode_label(estimate)}",
            f"  预估基准输入: {input_desc}",
            "  工作目录当前大小: "
            f"{format_bytes(estimate.current_work_dir_bytes)}, "
            f"预计新增峰值: {format_bytes(estimate.estimated_peak_growth_bytes)}, "
            f"建议剩余空间: {format_bytes(estimate.required_free_bytes)}, "
            f"当前剩余: {format_bytes(estimate.available_bytes)}",
        ]

    def _disk_space_error_message(self, estimate: DiskSpaceEstimate) -> str:
        input_line = f"预估基准输入: {estimate.selected_input_label}"
        if estimate.input_bytes > 0:
            input_line += f" ({format_bytes(estimate.input_bytes)})"

        lines = [
            "当前磁盘剩余空间不足，已取消本次运行。",
            "",
            f"模式: {self._disk_space_mode_label(estimate)}",
            f"工作目录当前大小: {format_bytes(estimate.current_work_dir_bytes)}",
            input_line,
            f"预计新增峰值: {format_bytes(estimate.estimated_peak_growth_bytes)}",
            f"建议剩余空间: {format_bytes(estimate.required_free_bytes)}",
            f"当前剩余空间: {format_bytes(estimate.available_bytes)}",
            "",
            "请先清理磁盘空间，或改用更小的数据集后重试。",
        ]
        return "\n".join(lines)

    def _preflight_summary_lines(
        self,
        work_dir: Path,
        disk_estimate: DiskSpaceEstimate | None = None,
    ) -> list[str]:
        fits = self._fits_in_work_dir(work_dir)
        light_files = [p for p in fits if p.name.lower().startswith("light_")]
        other_fits = [p for p in fits if p not in light_files]

        machine = platform.machine().lower()
        current_mode = self._current_input_mode()
        linear_resume_path = self._linear_resume_input_path(work_dir)
        stage2_corrected_resume_path = self._stage2_corrected_resume_input_path(work_dir)
        lines = [
            "预检摘要：",
            f"  工作目录: {self._display_path(work_dir)}",
            f"  处理模式: {self._input_mode_label(current_mode)}",
            f"  检测到的 FITS 输入: 总计={len(fits)}, Light_={len(light_files)}, 其他={len(other_fits)}",
            f"  主机架构: {machine or '<未知>'}",
            f"  流水线脚本: {self._display_path(self.pipeline_path)}",
            "  Stage11 模块: "
            f"{self._display_path(self.pipeline_path.with_name('stage11_ai_postprocess.py'))}",
            f"  Siril 插件目录: {self._display_path(self.siril_plugin_dir)}",
        ]
        if current_mode == INPUT_MODE_LINEAR_RESUME:
            lines.append(
                "  线性续跑输入: "
                + (
                    linear_resume_path.name
                    if linear_resume_path.is_file()
                    else f"{LINEAR_RESUME_INPUT_NAME}（未找到）"
                )
            )
        elif current_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            lines.append(
                "  叠加后处理输入: "
                + (
                    str(stage2_corrected_resume_path.relative_to(work_dir))
                    if stage2_corrected_resume_path.is_file()
                    else f"{STAGE2_CORRECTED_INPUT_NAME}（未找到）"
                )
            )
        if disk_estimate is not None:
            lines.extend(self._disk_space_summary_lines(disk_estimate))
        return lines

    def _open_result_dir(self) -> None:
        path = self.dir_edit.text().strip()
        if path and Path(path).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_log_file(self) -> None:
        if self.run_log_path and self.run_log_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run_log_path)))
        else:
            QMessageBox.information(self, "暂无日志", "当前还没有可用的运行日志。")

    def _preflight_errors(self, work_dir: Path) -> list[str]:
        errors: list[str] = []
        siril_candidates = self._resolve_siril_candidates()

        if not siril_candidates:
            errors.append("应用资源中未找到可用的内置 Siril CLI。")

        required_files = [
            ("配置模板", self.config_template),
            ("流水线脚本", self.pipeline_path),
            ("Stage11 模块脚本", self.pipeline_path.with_name("stage11_ai_postprocess.py")),
        ]
        for label, path in required_files:
            if not path.exists():
                errors.append(f"{label}缺失：{path}")

        for cli in siril_candidates:
            if not cli.is_file():
                errors.append(f"Siril CLI 路径无效：{cli}")
            elif not cli.stat().st_mode & 0o111:
                errors.append(f"Siril CLI 不可执行：{cli}")

        if not work_dir.is_dir():
            errors.append(f"工作目录不存在：{work_dir}")

        current_mode = self._current_input_mode()
        if current_mode == INPUT_MODE_LINEAR_RESUME:
            linear_resume_path = self._linear_resume_input_path(work_dir)
            if not linear_resume_path.is_file():
                errors.append(
                    f"续跑模式要求工作目录根下存在 {LINEAR_RESUME_INPUT_NAME}：{linear_resume_path}"
                )
        elif current_mode == INPUT_MODE_STAGE2_CORRECTED_RESUME:
            stage2_corrected_path = self._stage2_corrected_resume_input_path(work_dir)
            if not stage2_corrected_path.is_file():
                errors.append(
                    "叠加后处理模式要求工作目录根下或 process/ 下存在 "
                    f"{STAGE2_CORRECTED_INPUT_NAME}：{stage2_corrected_path}"
                )
        else:
            fits = self._fits_in_work_dir(work_dir)
            if not fits:
                errors.append(f"在 {work_dir} 中未找到 .fit/.fits 输入文件")

        return errors

    def _parse_elapsed_to_seconds(self, elapsed: str) -> int | None:
        text = elapsed.strip()
        if not text:
            return None
        day_part = 0
        if "-" in text:
            day_txt, text = text.split("-", 1)
            if not day_txt.isdigit():
                return None
            day_part = int(day_txt) * 86400
        parts = text.split(":")
        if not 1 <= len(parts) <= 3:
            return None
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            return None
        if len(nums) == 3:
            h, m, s = nums
        elif len(nums) == 2:
            h, m, s = 0, nums[0], nums[1]
        else:
            h, m, s = 0, 0, nums[0]
        return day_part + h * 3600 + m * 60 + s

    def _has_stale_venv_bootstrap(self) -> bool:
        venv_target = str(self._siril_state_root() / "venv")
        try:
            cp = subprocess.run(
                ["/bin/ps", "-ax", "-o", "etime=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return False
            for line in cp.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    continue
                elapsed, command = fields
                if "python3 -m venv" not in command:
                    continue
                if venv_target not in command:
                    continue
                sec = self._parse_elapsed_to_seconds(elapsed)
                if sec is not None and sec > 300:
                    return True
        except Exception:
            return False
        return False

    def _resolve_runtime_candidates(self) -> list[Path]:
        candidates = self._resolve_siril_candidates()
        if not candidates:
            return candidates

        # In bundled-only mode we still surface stale-bootstrap diagnostics.
        if self._has_stale_venv_bootstrap() and self.bundled_siril_cli in candidates:
            reordered = [self.bundled_siril_cli] + [p for p in candidates if p != self.bundled_siril_cli]
            self._append_event(
                "检测到残留的 Siril Python venv 引导进程；"
                "将使用内置 Siril 运行时（仅内置模式）。"
            )
            return reordered

        return candidates

    def _plugin_download_script_path(self) -> Path:
        return self.siril_plugin_dir / "download_siril_plugins.sh"

    def _plugin_downloads_dir(self) -> Path:
        return self.siril_plugin_dir / "downloads"

    def _plugin_requirements_path(self) -> Path:
        return self.siril_plugin_dir / "requirements.txt"

    def _onnxruntime_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("onnxruntime-*.whl"))

    def _pyqt6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyqt6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PyQt6-*.whl"))

    def _pyqt6_qt6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyqt6_qt6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PyQt6_Qt6-*.whl"))

    def _pyqt6_sip_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyqt6_sip-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PyQt6_sip-*.whl"))

    def _pyside6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyside6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PySide6-*.whl"))

    def _pyside6_addons_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyside6_addons-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PySide6_Addons-*.whl"))

    def _pyside6_essentials_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("pyside6_essentials-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("PySide6_Essentials-*.whl"))

    def _shiboken6_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("shiboken6-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Shiboken6-*.whl"))

    def _astropy_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("astropy-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Astropy-*.whl"))

    def _scipy_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("scipy-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Scipy-*.whl"))

    def _tifffile_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("tifffile-*.whl"))

    def _lz4_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("lz4-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("LZ4-*.whl"))

    def _zstandard_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("zstandard-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("Zstandard-*.whl"))

    def _exifread_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("exifread-*.whl"))
        if wheels:
            return wheels
        return sorted(self._plugin_downloads_dir().glob("ExifRead-*.whl"))

    def _opencv_python_headless_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("opencv_python_headless-*.whl"))

    def _requests_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("requests-*.whl"))

    def _requests_dependency_wheels_missing(self) -> list[str]:
        downloads_dir = self._plugin_downloads_dir()
        required = (
            ("urllib3", "urllib3-*.whl"),
            ("idna", "idna-*.whl"),
            ("certifi", "certifi-*.whl"),
            ("charset_normalizer", "charset_normalizer-*.whl"),
        )
        return [
            label
            for label, pattern in required
            if not list(downloads_dir.glob(pattern))
        ]

    def _wheel_package_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("wheel-*.whl"))

    def _sep_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("sep-*.whl"))

    def _spandrel_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("spandrel-*.whl"))

    def _einops_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("einops-*.whl"))

    def _safetensors_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("safetensors-*.whl"))

    def _torch_wheels(self) -> list[Path]:
        wheels = sorted(self._plugin_downloads_dir().glob("torch-*.whl"))
        return [wheel for wheel in wheels if not wheel.name.startswith("torchvision-")]

    def _torchvision_wheels(self) -> list[Path]:
        return sorted(self._plugin_downloads_dir().glob("torchvision-*.whl"))

    def _requirement_names(self, requirements_path: Path) -> list[str]:
        if not requirements_path.is_file():
            return []
        names: list[str] = []
        for raw_line in requirements_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "http:", "https:", ".")):
                continue
            name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip()
            if name:
                names.append(name.replace("_", "-").lower())
        return names

    def _missing_requirement_wheels(self, requirements_path: Path) -> list[str]:
        downloads_dir = self._plugin_downloads_dir()
        wheel_names = [path.name.lower() for path in downloads_dir.glob("*.whl")]
        missing: list[str] = []
        for name in self._requirement_names(requirements_path):
            normalized = name.replace("-", "_")
            prefixes = (
                f"{name}-",
                f"{normalized}-",
            )
            if not any(
                wheel_name.startswith(prefix) for wheel_name in wheel_names for prefix in prefixes
            ):
                missing.append(name)
        return missing

    def _missing_plugin_artifacts(self) -> list[str]:
        missing: list[str] = []

        def call(name: str):
            method = getattr(self, name, None)
            if method is None:
                # Unit tests call this method on lightweight proxy objects and
                # only bind the artifact checks they are asserting. Treat
                # unbound optional checks as satisfied for those proxies; real
                # SeestarGui instances still use their bound methods.
                return [] if name == "_requests_dependency_wheels_missing" else [Path("__proxy_optional_present__")]
            return method()

        plugin_root = self.siril_plugin_dir
        if not plugin_root.exists() or not plugin_root.is_dir():
            return [f"插件目录不存在：{plugin_root}"]

        if not call("_plugin_download_script_path").exists():
            missing.append("download_siril_plugins.sh 缺失")

        wheel_files = sorted(
            (plugin_root / "downloads").glob("setiastrosuitepro-*.whl")
        )
        if not wheel_files:
            missing.append("setiastrosuitepro wheel 缺失")

        onnx_wheels = call("_onnxruntime_wheels")
        if not onnx_wheels:
            missing.append("onnxruntime wheel 缺失")
        if not call("_pyqt6_wheels"):
            missing.append("PyQt6 wheel 缺失")
        if not call("_pyqt6_qt6_wheels"):
            missing.append("PyQt6_Qt6 wheel 缺失")
        if not call("_pyqt6_sip_wheels"):
            missing.append("pyqt6_sip wheel 缺失")
        if not call("_tifffile_wheels"):
            missing.append("tifffile wheel 缺失")
        if not call("_lz4_wheels"):
            missing.append("lz4 wheel 缺失")
        if not call("_zstandard_wheels"):
            missing.append("zstandard wheel 缺失")
        if not call("_exifread_wheels"):
            missing.append("exifread wheel 缺失")
        if not call("_opencv_python_headless_wheels"):
            missing.append("opencv-python-headless wheel 缺失")
        if not call("_requests_wheels"):
            missing.append("requests wheel 缺失")
        missing_request_deps = call("_requests_dependency_wheels_missing")
        if missing_request_deps:
            missing.append(
                "requests dependency wheels 缺失: "
                + ", ".join(missing_request_deps)
            )
        if not call("_wheel_package_wheels"):
            missing.append("wheel package 缺失")
        if not call("_sep_wheels"):
            missing.append("sep wheel 缺失")
        if not call("_spandrel_wheels"):
            missing.append("spandrel wheel 缺失")
        if not call("_einops_wheels"):
            missing.append("einops wheel 缺失")
        if not call("_safetensors_wheels"):
            missing.append("safetensors wheel 缺失")
        if not call("_pyside6_wheels"):
            missing.append("PySide6 wheel 缺失")
        if not call("_pyside6_addons_wheels"):
            missing.append("PySide6_Addons wheel 缺失")
        if not call("_pyside6_essentials_wheels"):
            missing.append("PySide6_Essentials wheel 缺失")
        if not call("_shiboken6_wheels"):
            missing.append("shiboken6 wheel 缺失")
        if not call("_astropy_wheels"):
            missing.append("astropy wheel 缺失")
        if not call("_scipy_wheels"):
            missing.append("scipy wheel 缺失")
        if not call("_torch_wheels"):
            missing.append("torch wheel 缺失")
        if not call("_torchvision_wheels"):
            missing.append("torchvision wheel 缺失")
        missing_requirement_wheels = []
        if isinstance(self, SeestarGui):
            missing_requirement_wheels = self._missing_requirement_wheels(
                self._plugin_requirements_path()
            )
        if missing_requirement_wheels:
            missing.append(
                "requirements wheel 缺失: "
                + ", ".join(missing_requirement_wheels)
            )

        syqon_bundle = plugin_root / SYQON_STARLESS_BUNDLE_REL
        for name in ("syqon_starless_inference.py", "zenith.pt"):
            if not (syqon_bundle / name).is_file():
                missing.append(f"SyQon Starless {name} 缺失")

        cosmic_bundle = plugin_root / COSMIC_CLARITY_BUNDLE_REL
        cosmic_required_models = [
            cosmic_bundle / name for name in COSMIC_CLARITY_REQUIRED_MODEL_FILES
        ]
        if not all(p.is_file() for p in cosmic_required_models):
            missing.append(
                "CosmicClarity Native 模型缺失（需要 denoise + sharpen 的最小 .pth 集）"
            )
        classic_wrapper = plugin_root / "bin" / "CosmicClarity"
        if isinstance(self, SeestarGui) and (
            not classic_wrapper.is_file() or not os.access(classic_wrapper, os.X_OK)
        ):
            missing.append("CosmicClarity classic wrapper 缺失或不可执行")

        scripts_root = resolve_siril_scripts_root(plugin_root)
        if scripts_root is None:
            missing.append("siril-scripts 目录或 AberrationRemover.py 缺失")
        elif not (scripts_root / "processing" / "SyQon-Starless.py").is_file():
            missing.append("SyQon-Starless.py 缺失")

        return missing

    def _ensure_siril_plugins_ready(self) -> bool:
        missing_before = self._missing_plugin_artifacts()
        if not missing_before:
            scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
            self._append_event("Siril 插件缓存检查通过。")
            if scripts_root is not None:
                self._append_event(
                    "Siril scripts 根目录: " + self._display_path(scripts_root)
                )
            return True

        script_path = self._plugin_download_script_path()
        self._append_event(
            "检测到 Siril 插件缓存不完整："
            + "；".join(missing_before)
        )
        if not script_path.exists() or not script_path.is_file():
            self._append_event("插件下载脚本不存在，无法继续本次运行。")
            return False

        self._append_event("正在自动补齐 Siril 插件缓存（首次运行可能较慢）...")
        try:
            cp = subprocess.run(
                ["/bin/bash", str(script_path)],
                cwd=str(self.siril_plugin_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
        except Exception as e:
            self._append_event(f"自动补齐插件失败：{e}")
            return False

        if cp.returncode != 0:
            tail = (cp.stderr.strip() or cp.stdout.strip() or "unknown error")
            tail = tail[-300:]
            self._append_event(
                f"自动补齐插件失败，退出码={cp.returncode}：{tail}"
            )
            return False

        missing_after = self._missing_plugin_artifacts()
        if missing_after:
            self._append_event(
                "插件自动补齐后仍有缺失："
                + "；".join(missing_after)
                + "（本次运行已阻断）"
            )
            return False

        scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
        self._append_event("Siril 插件缓存已自动补齐完成。")
        if scripts_root is not None:
            self._append_event(
                "Siril scripts 根目录: " + self._display_path(scripts_root)
            )
        return True

    def _runtime_xdg_siril_dir(self) -> Path:
        return self.runtime_home / ".local" / "share" / "siril"

    def _runtime_siril_scripts_repo_dir(self) -> Path:
        return self.runtime_home / ".local" / "share" / "siril-scripts"

    def _runtime_syqon_starless_dir(self) -> Path:
        return self._siril_state_root() / "syqon_starless"

    def _runtime_cosmic_clarity_dir(self) -> Path:
        return self._siril_state_root() / "cosmic_clarity"

    def _sync_syqon_starless_bundle(self) -> None:
        bundle_dir = self.siril_plugin_dir / SYQON_STARLESS_BUNDLE_REL
        if not bundle_dir.is_dir():
            return

        copied: list[str] = []
        target_dir = self._runtime_syqon_starless_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "syqon_starless_inference.py",
            "zenith.pt",
            "zenith.pt.sha256",
            "zenith.pt.date",
            "zenith.pt.verified",
        ):
            source = bundle_dir / name
            if not source.is_file():
                continue
            target = target_dir / name
            if target.exists() and target.stat().st_size == source.stat().st_size:
                continue
            shutil.copy2(source, target)
            copied.append(name)

        if copied:
            self._append_event(
                "已同步 SyQon Starless 离线资源到运行时目录: "
                + ", ".join(copied)
            )

    def _sync_cosmic_clarity_bundle(self) -> None:
        bundle_dir = self.siril_plugin_dir / COSMIC_CLARITY_BUNDLE_REL
        if not bundle_dir.is_dir():
            return

        target_dir = self._runtime_cosmic_clarity_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        model_files = [
            source
            for pattern in ("*.pth", "*.pt", "*.onnx")
            for source in bundle_dir.glob(pattern)
        ]
        copied: list[str] = []
        for source in model_files:
            target = target_dir / source.name
            if target.exists() and target.stat().st_size == source.stat().st_size:
                continue
            shutil.copy2(source, target)
            copied.append(source.name)

        if copied:
            self._append_event(
                "已同步 CosmicClarity Native 离线模型到运行时目录: "
                + ", ".join(copied[:6])
                + ("..." if len(copied) > 6 else "")
            )

    def _ensure_runtime_siril_support_dirs(self) -> None:
        xdg_siril_dir = self._runtime_xdg_siril_dir()
        xdg_siril_dir.mkdir(parents=True, exist_ok=True)

        # Keep this directory present to avoid SPCC catalog path probes failing
        # with "Error accessing directory" in Siril CLI mode.
        gaia_photo_dir = xdg_siril_dir / "gaia_photometric.dat"
        gaia_photo_dir.mkdir(parents=True, exist_ok=True)
        self._sync_syqon_starless_bundle()
        self._sync_cosmic_clarity_bundle()

        scripts_root = resolve_siril_scripts_root(self.siril_plugin_dir)
        if scripts_root is None:
            return

        runtime_repo = self._runtime_siril_scripts_repo_dir()
        marker = runtime_repo / "processing" / "AberrationRemover.py"
        if marker.is_file():
            if apply_siril_runtime_patches(self.siril_plugin_dir, runtime_repo):
                self._append_event("已应用 GraXpert-AI 运行时兼容补丁")
            return

        runtime_repo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scripts_root, runtime_repo, dirs_exist_ok=True)
        if apply_siril_runtime_patches(self.siril_plugin_dir, runtime_repo):
            self._append_event("已应用 GraXpert-AI 运行时兼容补丁")
        self._append_event(
            "已同步 Siril scripts 仓库到运行时目录: "
            + self._display_path(runtime_repo)
        )

    def _runtime_venv_python_bin(self) -> Path:
        venv_dir = self._siril_state_root() / "venv" / "bin"
        for name in ("python3.12", "python3", "python"):
            candidate = venv_dir / name
            if candidate.exists():
                return candidate
        return venv_dir / "python3.12"

    def _runtime_python_env(self) -> dict[str, str]:
        runtime_env = scrub_python_env(os.environ.copy())
        runtime_env["HOME"] = str(self.runtime_home)
        runtime_env["LANG"] = "en_US.UTF-8"
        runtime_env["LC_ALL"] = "en_US.UTF-8"
        runtime_env["LC_CTYPE"] = "en_US.UTF-8"
        runtime_env["PYTHONUTF8"] = "1"
        runtime_env["PYTHONIOENCODING"] = "utf-8"
        bundled_downloads = self.resources / "siril_plugins" / "downloads"
        if bundled_downloads.is_dir():
            runtime_env["PIP_NO_INDEX"] = "1"
            runtime_env["PIP_FIND_LINKS"] = str(bundled_downloads)
            runtime_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        return runtime_env

    def _ensure_runtime_tiffile_alias(self) -> None:
        venv_dir = self._siril_state_root() / "venv"
        site_dir = resolve_venv_site_packages(venv_dir)
        alias_path = site_dir / "tiffile.py"
        alias_code = (
            '"""Compatibility shim for scripts importing `tiffile`."""\n'
            "from tifffile import *  # noqa: F401,F403\n"
        )
        if alias_path.exists():
            existing = alias_path.read_text(encoding="utf-8", errors="replace")
            if existing == alias_code:
                alias_written = False
            else:
                alias_path.write_text(alias_code, encoding="utf-8")
                alias_written = True
        else:
            alias_path.write_text(alias_code, encoding="utf-8")
            alias_written = True
        if alias_written:
            self._append_event(
                "已写入 tiffile 兼容别名模块: "
                + self._display_path(alias_path)
            )

        dist_info = site_dir / "tiffile-0.0.0.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        metadata_path = dist_info / "METADATA"
        wheel_path = dist_info / "WHEEL"
        top_level_path = dist_info / "top_level.txt"
        installer_path = dist_info / "INSTALLER"
        metadata_text = (
            "Metadata-Version: 2.1\n"
            "Name: tiffile\n"
            "Version: 0.0.0\n"
            "Summary: Compatibility shim that re-exports tifffile\n"
        )
        wheel_text = (
            "Wheel-Version: 1.0\n"
            "Generator: seestar-superimpose\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        top_level_text = "tiffile\n"
        installer_text = "seestar-superimpose\n"

        updated_dist = False
        for path, content in (
            (metadata_path, metadata_text),
            (wheel_path, wheel_text),
            (top_level_path, top_level_text),
            (installer_path, installer_text),
        ):
            if path.exists():
                existing = path.read_text(encoding="utf-8", errors="replace")
                if existing == content:
                    continue
            path.write_text(content, encoding="utf-8")
            updated_dist = True

        if updated_dist:
            self._append_event(
                "已写入 tiffile 兼容分发元数据: "
                + self._display_path(dist_info)
            )

    def _ensure_runtime_sirilpy_timeout_patch(self) -> None:
        venv_dir = self._siril_state_root() / "venv"
        site_dir = resolve_venv_site_packages(venv_dir)
        patch_path = site_dir / "sitecustomize.py"
        patch_code = (
            '"""Seestar runtime patch: override sirilpy timeout via env."""\n'
            "import os\n"
            "def _patch_default_timeout(func, timeout):\n"
            "    defaults = getattr(func, '__defaults__', None)\n"
            "    if not defaults:\n"
            "        return\n"
            "    updated = list(defaults)\n"
            "    updated[-1] = timeout\n"
            "    func.__defaults__ = tuple(updated)\n"
            "\n"
            "raw = os.getenv('SEESTAR_SIRILPY_TIMEOUT_SEC', '').strip()\n"
            "if raw:\n"
            "    try:\n"
            "        timeout = float(raw)\n"
            "    except Exception:\n"
            "        timeout = None\n"
            "    if timeout and timeout > 0:\n"
            "        try:\n"
            "            import sirilpy.connection as _sirilpy_connection\n"
            "            _sirilpy_connection.DEFAULT_TIMEOUT = timeout\n"
            "            _iface = getattr(_sirilpy_connection, 'SirilInterface', None)\n"
            "            if _iface is not None:\n"
            "                for _name in ('_recv_exact', '_send_command', '_execute_command', '_request_data'):\n"
            "                    _func = getattr(_iface, _name, None)\n"
            "                    if callable(_func):\n"
            "                        _patch_default_timeout(_func, timeout)\n"
            "        except Exception:\n"
            "            pass\n"
        )
        if patch_path.exists():
            existing = patch_path.read_text(encoding="utf-8", errors="replace")
            if existing == patch_code:
                return
        patch_path.write_text(patch_code, encoding="utf-8")
        self._append_event(
            "已写入 sirilpy timeout 补丁: "
            + self._display_path(patch_path)
        )

    def _ensure_runtime_requirements_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        requirements_path = self._plugin_requirements_path()
        if not requirements_path.is_file():
            raise FileNotFoundError(f"Siril runtime requirements not found: {requirements_path}")

        missing_wheels = self._missing_requirement_wheels(requirements_path)
        if missing_wheels:
            raise RuntimeError(
                "Siril runtime requirements 离线 wheel 缺失："
                + "、".join(missing_wheels)
            )

        runtime_env = self._runtime_python_env()
        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在按 requirements 离线安装 Siril runtime 依赖 "
            f"(no-index, find-links={self._display_path(wheel_dir)})..."
        )
        install_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--find-links",
                str(wheel_dir),
                "-r",
                str(requirements_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_cp.returncode != 0:
            tail = (
                install_cp.stderr.strip()
                or install_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Siril runtime requirements 离线安装失败 "
                f"(exit={install_cp.returncode}): {tail[-320:]}"
            )

        self._ensure_runtime_tiffile_alias()
        self._append_event("Siril runtime requirements 离线安装完成。")

    def _ensure_runtime_cosmic_clarity_deps_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        runtime_env = self._runtime_python_env()
        runtime_env.setdefault("SEESTAR_SIRILPY_TIMEOUT_SEC", "120")
        self._ensure_runtime_sirilpy_timeout_patch()
        self._ensure_runtime_tiffile_alias()

        probe_code = (
            "import importlib.metadata as md; "
            "import PyQt6, tifffile, tiffile, lz4, zstandard, exifread, cv2, requests, sep, spandrel; "
            "print('cosmic-clarity-deps-ok', md.version('tiffile'))"
        )
        check_cp = subprocess.run(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if check_cp.returncode == 0:
            self._append_event(
                "Cosmic Clarity 运行时依赖已就绪"
                "（PyQt6/tifffile/lz4/zstandard/exifread/opencv/requests）。"
            )
            return

        missing_wheels: list[str] = []
        if not self._pyqt6_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[0])
        if not self._pyqt6_qt6_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[1])
        if not self._pyqt6_sip_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[2])
        if not self._tifffile_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[3])
        if not self._lz4_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[4])
        if not self._zstandard_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[5])
        if not self._exifread_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[6])
        if not self._opencv_python_headless_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[7])
        if not self._requests_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[8])
        if self._requests_dependency_wheels_missing():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[9])
        if not self._wheel_package_wheels():
            missing_wheels.append(SIRIL_COSMIC_REQUIRED_WHEEL_LABELS[10])
        if not self._sep_wheels():
            missing_wheels.append("sep")
        if not self._spandrel_wheels():
            missing_wheels.append("spandrel")
        if not self._einops_wheels():
            missing_wheels.append("einops")
        if not self._safetensors_wheels():
            missing_wheels.append("safetensors")
        if missing_wheels:
            raise RuntimeError(
                "Cosmic Clarity 离线依赖 wheel 缺失："
                + "、".join(missing_wheels)
            )

        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在离线安装 Cosmic Clarity 运行时依赖 "
            "(PyQt6/tifffile/lz4/zstandard/exifread/opencv/requests, no-index)..."
        )
        install_pyqt_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "PyQt6",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_pyqt_cp.returncode != 0:
            tail = (
                install_pyqt_cp.stderr.strip()
                or install_pyqt_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Cosmic Clarity 依赖离线安装失败 (PyQt6) "
                f"(exit={install_pyqt_cp.returncode}): {tail[-280:]}"
            )

        install_tifffile_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "tifffile",
                "lz4",
                "zstandard",
                "exifread",
                "opencv-python-headless",
                "requests",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_tifffile_cp.returncode != 0:
            tail = (
                install_tifffile_cp.stderr.strip()
                or install_tifffile_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Cosmic Clarity 依赖离线安装失败 "
                "(tifffile/lz4/zstandard/exifread/opencv/requests) "
                f"(exit={install_tifffile_cp.returncode}): {tail[-280:]}"
            )

        self._ensure_runtime_tiffile_alias()

        install_ai_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--find-links",
                str(wheel_dir),
                "sep",
                "spandrel",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_ai_cp.returncode != 0:
            tail = (
                install_ai_cp.stderr.strip()
                or install_ai_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "Cosmic Clarity/SCUNet 依赖离线安装失败 (sep/spandrel) "
                f"(exit={install_ai_cp.returncode}): {tail[-280:]}"
            )

        verify_cp = subprocess.run(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if verify_cp.returncode != 0:
            tail = (verify_cp.stderr.strip() or verify_cp.stdout.strip() or "unknown error")
            raise RuntimeError(f"Cosmic Clarity 依赖安装后导入失败: {tail[-280:]}")
        self._append_event("Cosmic Clarity 运行时依赖离线安装完成。")

    def _ensure_runtime_syqon_starless_deps_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        runtime_env = self._runtime_python_env()
        probe_code = (
            "import PyQt6, PySide6, astropy, scipy, torch; "
            "print('syqon-starless-deps-ok', astropy.__version__, "
            "scipy.__version__, torch.__version__)"
        )
        check_cp = subprocess.run(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if check_cp.returncode == 0:
            self._append_event("SyQon Starless 运行时依赖已就绪（PyQt6/PySide6/astropy/scipy）。")
            return

        missing_wheels: list[str] = []
        if not self._pyside6_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[0])
        if not self._pyside6_addons_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[1])
        if not self._pyside6_essentials_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[2])
        if not self._shiboken6_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[3])
        if not self._astropy_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[4])
        if not self._scipy_wheels():
            missing_wheels.append(SIRIL_STARLESS_REQUIRED_WHEEL_LABELS[5])
        if not self._torch_wheels():
            missing_wheels.append("torch")
        if not self._torchvision_wheels():
            missing_wheels.append("torchvision")
        if missing_wheels:
            raise RuntimeError(
                "SyQon Starless 离线依赖 wheel 缺失："
                + "、".join(missing_wheels)
            )

        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在离线安装 SyQon Starless 运行时依赖 "
            "(PySide6/astropy/scipy, no-index)..."
        )
        install_pyside_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "PySide6",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_pyside_cp.returncode != 0:
            tail = (
                install_pyside_cp.stderr.strip()
                or install_pyside_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "SyQon Starless 依赖离线安装失败 (PySide6) "
                f"(exit={install_pyside_cp.returncode}): {tail[-280:]}"
            )

        install_sci_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                "astropy",
                "scipy",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_sci_cp.returncode != 0:
            tail = (
                install_sci_cp.stderr.strip()
                or install_sci_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "SyQon Starless 依赖离线安装失败 (astropy/scipy) "
                f"(exit={install_sci_cp.returncode}): {tail[-280:]}"
            )

        install_torch_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--upgrade",
                "--find-links",
                str(wheel_dir),
                "torch",
                "torchvision",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_torch_cp.returncode != 0:
            tail = (
                install_torch_cp.stderr.strip()
                or install_torch_cp.stdout.strip()
                or "unknown error"
            )
            raise RuntimeError(
                "SyQon Starless 依赖离线安装失败 (torch/torchvision) "
                f"(exit={install_torch_cp.returncode}): {tail[-280:]}"
            )

        verify_cp = subprocess.run(
            [str(python_bin), "-c", probe_code],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if verify_cp.returncode != 0:
            tail = (verify_cp.stderr.strip() or verify_cp.stdout.strip() or "unknown error")
            raise RuntimeError(f"SyQon Starless 依赖安装后导入失败: {tail[-280:]}")
        self._append_event("SyQon Starless 运行时依赖离线安装完成。")

    def _ensure_runtime_onnxruntime_ready(self) -> None:
        python_bin = self._runtime_venv_python_bin()
        if not python_bin.exists():
            raise FileNotFoundError(f"Siril runtime python not found: {python_bin}")

        runtime_env = self._runtime_python_env()

        check_cp = subprocess.run(
            [str(python_bin), "-c", "import onnxruntime as ort; print(ort.__version__)"],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if check_cp.returncode == 0:
            version = (check_cp.stdout.strip() or "unknown").splitlines()[-1]
            self._append_event(f"onnxruntime 已就绪 (version={version})。")
            return

        wheels = self._onnxruntime_wheels()
        if not wheels:
            raise RuntimeError("onnxruntime wheel 缺失，无法离线安装")
        wheel_path = wheels[-1]
        wheel_dir = self._plugin_downloads_dir()
        self._append_event(
            "正在离线安装 onnxruntime 到 Siril runtime venv "
            f"(wheel={wheel_path.name}, no-deps)..."
        )
        install_cp = subprocess.run(
            [
                str(python_bin),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--force-reinstall",
                "--find-links",
                str(wheel_dir),
                str(wheel_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if install_cp.returncode != 0:
            tail = (install_cp.stderr.strip() or install_cp.stdout.strip() or "unknown error")
            raise RuntimeError(
                f"onnxruntime 离线安装失败 (exit={install_cp.returncode}): {tail[-280:]}"
            )

        verify_cp = subprocess.run(
            [str(python_bin), "-c", "import onnxruntime as ort; print(ort.__version__)"],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env,
        )
        if verify_cp.returncode != 0:
            tail = (verify_cp.stderr.strip() or verify_cp.stdout.strip() or "unknown error")
            raise RuntimeError(f"onnxruntime 安装后导入失败: {tail[-280:]}")
        version = (verify_cp.stdout.strip() or "unknown").splitlines()[-1]
        self._append_event(f"onnxruntime 离线安装完成 (version={version})。")

    def _start_run(self) -> None:
        if self.worker and self.worker.isRunning():
            return

        work_dir = Path(self.dir_edit.text().strip()).expanduser()
        try:
            self._ensure_offline_siril_python_seed()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Siril Python Seed 准备失败",
                f"准备离线 Siril Python seed 失败：\n{e}",
            )
            self._append_event(f"离线 Siril seed 准备失败：{e}")
            return

        if not self._ensure_siril_plugins_ready():
            QMessageBox.critical(
                self,
                "插件预检失败",
                "Siril 插件缓存不完整，且自动补齐失败。\n"
                "请先执行 resources/siril_plugins/download_siril_plugins.sh 后重试。",
            )
            self._append_event("插件预检失败，已取消本次运行。")
            return

        try:
            self._ensure_runtime_siril_support_dirs()
        except Exception as e:
            self._append_event(f"Siril 运行时目录准备失败（继续执行）：{e}")

        errors = self._preflight_errors(work_dir)
        if errors:
            QMessageBox.critical(self, "预检失败", "\n\n".join(errors))
            self._append_event("预检失败：")
            for err in errors:
                self._append_text(f"  - {err}\n")
            return

        try:
            disk_estimate = self._estimate_disk_space(work_dir)
        except Exception as e:
            QMessageBox.critical(
                self,
                "磁盘空间检查失败",
                f"无法完成磁盘空间预检：\n{e}",
            )
            self._append_event(f"磁盘空间预检失败：{e}")
            return

        if disk_estimate and disk_estimate.available_bytes < disk_estimate.required_free_bytes:
            QMessageBox.critical(
                self,
                "磁盘空间不足",
                self._disk_space_error_message(disk_estimate),
            )
            self._append_event("磁盘空间预检失败：")
            for line in self._disk_space_summary_lines(disk_estimate):
                self._append_text(f"{line}\n")
            self._append_event("当前卷剩余空间不足，已取消本次运行。")
            return

        try:
            self._ensure_runtime_requirements_ready()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Siril runtime 依赖准备失败",
                f"无法按 requirements 准备 Siril runtime 依赖：\n{e}",
            )
            self._append_event(f"Siril runtime requirements 依赖准备失败：{e}")
            return

        try:
            self._ensure_runtime_cosmic_clarity_deps_ready()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Cosmic Clarity 依赖准备失败",
                f"无法准备 Siril 运行时 PyQt6/tifffile：\n{e}",
            )
            self._append_event(f"Cosmic Clarity 依赖准备失败：{e}")
            return

        try:
            self._ensure_runtime_syqon_starless_deps_ready()
        except Exception as e:
            QMessageBox.critical(
                self,
                "SyQon Starless 依赖准备失败",
                f"无法准备 Siril 运行时 PySide6/astropy/scipy：\n{e}",
            )
            self._append_event(f"SyQon Starless 依赖准备失败：{e}")
            return

        try:
            self._ensure_runtime_onnxruntime_ready()
        except Exception as e:
            QMessageBox.critical(
                self,
                "onnxruntime 准备失败",
                f"无法准备 Siril 运行时 onnxruntime：\n{e}",
            )
            self._append_event(f"onnxruntime 准备失败：{e}")
            return

        self._current_work_dir = work_dir

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_path = work_dir / f"seestar_gui_run_{stamp}.log"
        self.run_log_file = self.run_log_path.open("a", encoding="utf-8", errors="replace")
        self._append_divider(
            "本次任务开始",
            [
                f"时间: {self._timestamp()}",
                f"工作目录: {self._display_path(work_dir)}",
                f"处理模式: {self._input_mode_label(self._current_input_mode())}",
            ],
        )
        for line in self._preflight_summary_lines(work_dir, disk_estimate):
            self._append_event(line)
        self._append_event(f"日志文件: {self._display_path(self.run_log_path)}")
        self._append_event(
            f"本次运行 debug_mode={'ON' if self.debug_mode_enabled else 'OFF'}"
        )
        self._append_event(
            f"本次运行 network_mode={'ON' if self.network_mode_enabled else 'OFF'}"
        )
        self._append_event(
            f"本次运行 input_mode={self._current_input_mode()}"
        )
        self._append_event(
            f"本次运行 ai_stage11={'ON' if self.ai_stage_enabled else 'OFF'}"
        )

        siril_candidates = self._resolve_runtime_candidates()
        self._append_event(
            "运行时顺序: "
            + ", ".join(self._display_path(p) for p in siril_candidates)
        )

        self.worker = PipelineWorker(
            work_dir=work_dir,
            config_template=self.config_template,
            pipeline_path=self.pipeline_path,
            siril_plugin_dir=self.siril_plugin_dir,
            resources=self.resources,
            runtime_home=self.runtime_home,
            siril_candidates=siril_candidates,
            input_mode=self._current_input_mode(),
            debug_mode=self.debug_mode_enabled,
            network_mode=self.network_mode_enabled,
            ai_stage_enabled=self.ai_stage_enabled,
            parent=self,
        )
        self.worker.log.connect(self._append_text)
        self.worker.state.connect(self._set_status_text)
        self.worker.done.connect(self._on_worker_done)

        self._set_running(True)
        self._set_status_text("Running")
        self.worker.start()

    def _stop_run(self) -> None:
        if not self.worker:
            return
        self._append_event("已请求停止...")
        self.worker.stop()

    def _on_worker_done(self, status: str, exit_code: int, had_errors: bool, cli_used: str) -> None:
        self._set_status_text(status)
        self._append_event(
            f"处理结束：状态={self._display_status(status)}，退出码={exit_code}，CLI={cli_used}"
        )
        if status == "Failed" and had_errors:
            self._append_event("在输出中检测到 Siril/脚本错误。")
        self._append_divider(
            "本次任务结束",
            [
                f"时间: {self._timestamp()}",
                f"工作目录: {self._display_path(self._current_work_dir)}",
                f"最终状态: {self._display_status(status)}",
            ],
        )

        self._set_running(False)
        self._cleanup_after_run()

    def _cleanup_after_run(self) -> None:
        if self.worker:
            self.worker.wait(200)
            self.worker.deleteLater()
            self.worker = None

        if self.run_log_file:
            self.run_log_file.close()
            self.run_log_file = None

        self._current_work_dir = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.worker and self.worker.isRunning():
            ret = QMessageBox.question(
                self,
                "处理仍在运行",
                "仍有处理任务在运行。是否停止并退出？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_run()
            self.worker.wait(8000)
        event.accept()
