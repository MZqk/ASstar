#!/usr/bin/env python3
"""Seestar Superimpose macOS GUI launcher with external pipeline resource."""

from __future__ import annotations

import importlib.util
import os
import platform
import queue
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from math import floor
from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QTextCursor
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
FITS_SUFFIXES = (".fit", ".fits")
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
LIGHT_FRAME_EXPANSION_FACTOR = 3.0
LIGHT_PREPROCESS_SEQUENCE_COPIES = 2.0
STACKED_STAGE_ARTIFACT_COPIES = 12.0
LINEAR_RESUME_STAGE_ARTIFACT_COPIES = 8.0
DISK_SPACE_HEADROOM_RATIO = 0.15
DISK_SPACE_MIN_HEADROOM_BYTES = 1 * 1024 * 1024 * 1024


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


@dataclass(frozen=True)
class DiskSpaceEstimate:
    mode: str
    current_work_dir_bytes: int
    input_count: int
    input_bytes: int
    estimated_peak_growth_bytes: int
    required_free_bytes: int
    available_bytes: int
    selected_input_label: str


def format_bytes(num_bytes: int) -> str:
    value = float(max(num_bytes, 0))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if unit == units[-1] or value < 1024.0:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{int(value)} B"


def safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.is_dir():
        return total

    for root, _dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                if file_path.is_symlink():
                    continue
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


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


class PipelineWorker(QThread):
    """Runs Siril processing in a worker thread via siril-cli subprocess."""

    log = Signal(str)
    state = Signal(str)
    done = Signal(str, int, bool, str)

    def __init__(
        self,
        work_dir: Path,
        config_template: Path,
        pipeline_path: Path,
        siril_plugin_dir: Path,
        resources: Path,
        runtime_home: Path,
        siril_candidates: list[Path],
        input_mode: str = INPUT_MODE_AUTO,
        debug_mode: bool = False,
        network_mode: bool = True,
        ai_stage_enabled: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.work_dir = work_dir
        self.config_template = config_template
        self.pipeline_path = pipeline_path
        self.siril_plugin_dir = siril_plugin_dir
        self.resources = resources
        self.runtime_home = runtime_home
        self.siril_candidates = siril_candidates
        self.input_mode = (
            input_mode
            if input_mode in {
                INPUT_MODE_AUTO,
                INPUT_MODE_LINEAR_RESUME,
                INPUT_MODE_STAGE2_CORRECTED_RESUME,
            }
            else INPUT_MODE_AUTO
        )
        self.debug_mode = bool(debug_mode)
        self.network_mode = bool(network_mode)
        self.ai_stage_enabled = bool(ai_stage_enabled)

        self._stop_event = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._active_mode = "python"
        self._run_had_errors = False
        self._last_output_ts = 0.0
        self._pyscript_seen_at: float | None = None
        self._pipeline_output_seen = False
        self._python_env_issue = False
        self._python_env_repair_attempted = False
        self._spcc_seen_in_run = False
        self._spcc_cli_crash_detected = False
        self._spcc_crash_retry_attempted = False
        self._force_disable_spcc_for_retry = False
        self._recent_process_output: deque[str] = deque(maxlen=80)
        self._last_spcc_command = ""
        self._ai_env_sources: list[str] = []
        self._ai_env_applied_keys: list[str] = []
        self._ai_env_warnings: list[str] = []
        self._runtime_plugin_dir: Path | None = None

    def stop(self) -> None:
        self._stop_event.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_event(self, msg: str) -> None:
        self.log.emit(f"[{self._timestamp()}] {msg}\n")

    def _ai_env_candidates(self) -> list[Path]:
        return [
            self.resources / DEFAULT_ENV_RESOURCE_REL,
            self.resources / AI_ENV_RESOURCE_REL,
            self.runtime_home / AI_ENV_OVERRIDE_NAME,
            self.work_dir / AI_ENV_OVERRIDE_NAME,
        ]

    def _load_ai_env_overrides(self) -> tuple[dict[str, str], list[str], list[str]]:
        merged: dict[str, str] = {}
        sources: list[str] = []
        warnings: list[str] = []

        for path in self._ai_env_candidates():
            if not path.exists() or not path.is_file():
                continue
            parsed, parse_warnings = parse_ai_env_file(path)
            sources.append(str(path))
            merged.update(parsed)
            warnings.extend(parse_warnings)

        return merged, sources, warnings

    def _inspect_output_for_errors(self, text: str) -> None:
        lowered = text.lower()
        stripped = text.strip()
        if stripped:
            self._recent_process_output.append(stripped)
        error_markers = (
            "script execution failed",
            "failed to install python module",
            "python not ready yet",
            "python validation failed",
            "python version check failed",
            "failed to initialize python virtual environment",
            "unable to spawn python",
            "error in line",
            "unknown error",
            "exiting batch processing",
        )
        python_env_markers = (
            "failed to install python module",
            "python not ready yet",
            "unable to install or update the siril python module",
            "failed to execute pip",
            "pip command failed",
            "python validation failed",
            "python version check failed",
            "failed to initialize python virtual environment",
            "error finding venv python path",
            "unable to spawn python",
            "failed to create python connection",
            "failed to execute python script",
        )
        if any(marker in lowered for marker in error_markers):
            self._run_had_errors = True
        if any(marker in lowered for marker in python_env_markers):
            self._python_env_issue = True
            # Treat Python environment problems as hard pipeline errors even when
            # siril-cli exits 0, so caller can retry/reset/fallback correctly.
            self._run_had_errors = True
        if (
            "running command: spcc" in lowered
            or "running command spcc" in lowered
            or "input command:spcc" in lowered
            or "input command: spcc" in lowered
        ):
            self._spcc_seen_in_run = True
            if "spcc" in lowered:
                self._last_spcc_command = stripped
        if (
            ("running command: pyscript" in lowered or "running command pyscript" in lowered)
            and self._pyscript_seen_at is None
        ):
            self._pyscript_seen_at = time.time()
        if self._pyscript_seen_at is not None and stripped:
            # Any non-empty output after entering pyscript means the pipeline is alive.
            if "running command: pyscript" not in lowered and "running command pyscript" not in lowered:
                self._pipeline_output_seen = True
        pipeline_markers = ("stage 1", "阶段 1", "[info]")
        if any(marker in lowered for marker in pipeline_markers):
            self._pipeline_output_seen = True

    def _append_spcc_crash_diagnostics(self, exit_code: int) -> None:
        self._append_event(
            "SPCC 崩溃诊断: siril-cli 在执行 SPCC 后以 "
            f"{exit_code} 退出，通常表示 Siril 原生测光/SPCC 代码段错误，"
            "Python 侧不会产生 CommandError。"
        )
        if self._last_spcc_command:
            self._append_event(f"SPCC 崩溃前命令标记: {self._last_spcc_command}")
        if self._recent_process_output:
            self._append_event("SPCC 崩溃前最后输出（最多 25 行）:")
            for line in list(self._recent_process_output)[-25:]:
                self._append_event(f"  {line}")

    def _siril_venv_dir(self) -> Path:
        return self._siril_state_root() / "venv"

    def _siril_state_root(self) -> Path:
        return siril_state_root_from_home(self.runtime_home)

    def _restore_offline_siril_seed(self) -> bool:
        state_root = self._siril_state_root()
        venv_dir = state_root / "venv"
        module_dir = state_root / ".python_module"
        seed_root = self.resources / "SirilPythonSeed"
        seed_venv = seed_root / "venv"
        seed_module = seed_root / ".python_module"

        if not seed_venv.exists() or not seed_module.exists():
            self._append_event(
                f"应用资源中缺少离线 Siril seed：{seed_root}"
            )
            return False

        try:
            state_root.mkdir(parents=True, exist_ok=True)
            if not venv_dir.exists():
                shutil.copytree(seed_venv, venv_dir, symlinks=True)
                self._append_event(f"已从离线 seed 恢复 Siril venv：{venv_dir}")
            if not module_dir.exists():
                shutil.copytree(seed_module, module_dir, symlinks=True)
                self._append_event(
                    f"已从离线 seed 恢复 Siril Python 模块：{module_dir}"
                )
            elif not (module_dir / "sirilpy").exists():
                shutil.rmtree(module_dir, ignore_errors=True)
                shutil.copytree(seed_module, module_dir, symlinks=True)
                self._append_event(
                    f"已重新从离线 seed 恢复 Siril Python 模块：{module_dir}"
                )

            # Rewrite venv interpreter links/config to current bundled Siril path.
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
                self._append_event(f"内置 Siril Python 缺失：{py_bin}")
                return False

            bin_dir = venv_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for name in ("python3.12", "python3", "python"):
                dst = bin_dir / name
                if dst.exists() or dst.is_symlink():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                dst.symlink_to(py_bin)

            cfg = venv_dir / "pyvenv.cfg"
            cfg_lines: list[str] = []
            if cfg.exists():
                cfg_lines = cfg.read_text(encoding="utf-8", errors="replace").splitlines()
            siril_python = self.resources / "Siril.app" / "Contents" / "MacOS" / "python3"
            replacements = {
                "home": str(py_bin.parent),
                "executable": str(py_bin),
                "command": f"{siril_python} -m venv {venv_dir}",
            }
            seen: set[str] = set()
            out_lines: list[str] = []
            for line in cfg_lines:
                if "=" not in line:
                    out_lines.append(line)
                    continue
                key, _value = line.split("=", 1)
                k = key.strip()
                if k in replacements:
                    out_lines.append(f"{k} = {replacements[k]}")
                    seen.add(k)
                else:
                    out_lines.append(line)
            for k, v in replacements.items():
                if k not in seen:
                    out_lines.append(f"{k} = {v}")
            cfg.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

            site_dir = resolve_venv_site_packages(venv_dir)
            repaired = repair_site_packages_from_pip_vendor(site_dir)
            if repaired:
                self._append_event(
                    "已从 pip vendor 修补 Siril venv 依赖："
                    + ", ".join(repaired)
                )
            ok, detail = verify_siril_offline_seed_venv(venv_dir)
            if not ok:
                self._append_event(f"离线 Siril seed 校验失败：{detail}")
                return False
            self._append_event(f"离线 Siril seed 校验通过：{detail}")

            return True
        except Exception as e:
            self._append_event(f"恢复离线 Siril seed 失败：{e}")
            return False

    def _collect_processes(self, needle_parts: tuple[str, ...]) -> list[tuple[int, str]]:
        matches: list[tuple[int, str]] = []
        try:
            cp = subprocess.run(
                ["/bin/ps", "-ax", "-o", "pid=,command="],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode != 0:
                return matches
            for line in cp.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    continue
                pid_txt, command = fields
                if not pid_txt.isdigit():
                    continue
                pid = int(pid_txt)
                if pid <= 0 or pid == os.getpid():
                    continue
                if all(part in command for part in needle_parts):
                    matches.append((pid, command))
        except Exception:
            return matches
        return matches

    def _terminate_processes(self, procs: list[tuple[int, str]]) -> int:
        if not procs:
            return 0
        terminated = 0
        for pid, _command in procs:
            try:
                os.kill(pid, signal.SIGTERM)
                terminated += 1
            except Exception:
                continue
        time.sleep(1.0)
        for pid, _command in procs:
            try:
                os.kill(pid, 0)
            except Exception:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                continue
        return terminated

    def _reset_siril_python_venv(self) -> bool:
        venv_dir = self._siril_venv_dir()
        python_token = str(venv_dir / "bin/python")
        pip_token = str(venv_dir / "bin/pip")
        venv_target = str(venv_dir)

        # Terminate stale venv python/pip processes first.
        stale = self._collect_processes((python_token,))
        stale += self._collect_processes((pip_token,))
        stale += self._collect_processes(("python3 -m venv", venv_target))
        if stale:
            terminated = self._terminate_processes(stale)
            self._append_event(
                f"已终止 {terminated} 个残留的 Siril Python/venv 进程。"
            )

        if not venv_dir.exists():
            self._append_event(
                "Siril Python venv 不存在，开始恢复离线 seed。"
            )
            return self._restore_offline_siril_seed()

        try:
            shutil.rmtree(venv_dir)
            self._append_event(f"已删除 Siril Python venv：{venv_dir}")
            return self._restore_offline_siril_seed()
        except Exception as e:
            self._append_event(f"删除 Siril Python venv 失败：{e}")
            return False

    def _reader(self, stream, out_queue: queue.Queue[str | None]) -> None:
        try:
            for line in iter(stream.readline, ""):
                out_queue.put(line)
        finally:
            out_queue.put(None)

    def _prepare_runtime_files(self, temp_dir: Path) -> tuple[Path, Path, Path]:
        run_ssf = temp_dir / "run_job_embedded.ssf"
        run_ini = temp_dir / "config.1.4.ini"
        run_py = temp_dir / self.pipeline_path.name
        pipeline_dir = self.pipeline_path.parent
        stage11_module_path = self.pipeline_path.with_name("stage11_ai_postprocess.py")
        run_stage11_module = temp_dir / stage11_module_path.name
        cpu_limit = compute_siril_cpu_limit()

        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"未找到流水线脚本：{self.pipeline_path}")
        for py_file in pipeline_dir.glob("*.py"):
            shutil.copy2(py_file, temp_dir / py_file.name)
        stages_dir = pipeline_dir / "stages"
        if stages_dir.exists() and stages_dir.is_dir():
            shutil.copytree(stages_dir, temp_dir / "stages", dirs_exist_ok=True)
        if not stage11_module_path.exists():
            raise FileNotFoundError(f"未找到 Stage11 模块脚本：{stage11_module_path}")
        if not run_stage11_module.exists():
            shutil.copy2(stage11_module_path, run_stage11_module)

        self._runtime_plugin_dir = None
        if self.siril_plugin_dir.exists() and self.siril_plugin_dir.is_dir():
            plugin_dst = temp_dir / "siril_plugins"
            shutil.copytree(self.siril_plugin_dir, plugin_dst, dirs_exist_ok=True)
            if apply_siril_runtime_patches(plugin_dst):
                self._append_event("已应用 GraXpert-AI 运行时兼容补丁")
            self._runtime_plugin_dir = plugin_dst

        ssf_lines = [
            "requires 1.4.0",
        ]
        if cpu_limit is not None:
            ssf_lines.append(f"setcpu {cpu_limit}")
        ssf_lines.extend(
            [
                f'cd "{shell_quote_path(self.work_dir)}"',
                f'pyscript "{shell_quote_path(run_py)}"',
                "close",
            ]
        )
        run_ssf.write_text("\n".join(ssf_lines) + "\n", encoding="utf-8")

        template_text = self.config_template.read_text(
            encoding="utf-8", errors="replace"
        )
        patched = normalize_siril_config_template(template_text)
        run_ini.write_text(patched, encoding="utf-8")
        return run_ssf, run_ini, run_py

    def _build_env(self, siril_cli: Path) -> dict[str, str]:
        env = scrub_python_env(os.environ.copy())
        ai_env, ai_sources, ai_warnings = self._load_ai_env_overrides()
        applied_keys: list[str] = []
        for key, value in ai_env.items():
            if not env.get(key):
                env[key] = value
                applied_keys.append(key)
        self._ai_env_sources = ai_sources
        self._ai_env_applied_keys = sorted(applied_keys)
        self._ai_env_warnings = ai_warnings

        # Finder-launched apps may lack UTF-8 locale vars.
        env["HOME"] = str(self.runtime_home)
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"
        env["LC_CTYPE"] = "en_US.UTF-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("SEESTAR_SIRILPY_TIMEOUT_SEC", "120")
        env["PIP_NO_INDEX"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        pip_find_links: list[str] = []
        bundled_downloads = self.resources / "siril_plugins" / "downloads"
        if bundled_downloads.is_dir():
            pip_find_links.append(str(bundled_downloads))
        if self._runtime_plugin_dir:
            runtime_downloads = self._runtime_plugin_dir / "downloads"
            if runtime_downloads.is_dir():
                pip_find_links.append(str(runtime_downloads))
        if pip_find_links:
            env["PIP_FIND_LINKS"] = " ".join(dict.fromkeys(pip_find_links))
        bundled_py = (
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
        if bundled_py.exists():
            env["SIRIL_PYTHON_CLI"] = str(bundled_py)
            env["SEESTAR_SIRIL_PYTHON_CLI"] = str(bundled_py)

        bundled_siril_cli = self.resources / "Siril.app" / "Contents" / "MacOS" / "siril-cli"
        if siril_cli == bundled_siril_cli:
            relocated = self.resources / "Siril.app" / "Contents" / "Resources"
            env["SIRIL_RELOCATED_RES_DIR"] = str(relocated)

        if self._runtime_plugin_dir and self._runtime_plugin_dir.exists():
            env["SEESTAR_SIRIL_PLUGIN_DIR"] = str(self._runtime_plugin_dir)
            classic_wrapper = self._runtime_plugin_dir / "bin" / "CosmicClarity"
            if classic_wrapper.is_file() and os.access(classic_wrapper, os.X_OK):
                env.setdefault("SEESTAR_COSMIC_CLARITY_EXECUTABLE", str(classic_wrapper))
            scripts_dir = resolve_siril_scripts_root(self._runtime_plugin_dir)
            if scripts_dir is not None:
                env["SIRIL_SCRIPTS_DIR"] = str(scripts_dir)
                env["SIRIL_SCRIPTS_PATH"] = str(scripts_dir)

        env["SEESTAR_DEBUG_MODE"] = "1" if self.debug_mode else "0"
        env["SEESTAR_INPUT_MODE"] = self.input_mode
        # GUI toggle is the highest-priority control for optional stage11 execution.
        env["SEESTAR_AI_ENABLED"] = "1" if self.ai_stage_enabled else "0"
        if self._force_disable_spcc_for_retry:
            env["SEESTAR_SPCC_ENABLE"] = "0"
        return env

    def _run_once(self, siril_cli: Path, run_ssf: Path, run_ini: Path) -> tuple[bool, int]:
        self._run_had_errors = False
        self._python_env_issue = False
        self._pyscript_seen_at = None
        self._pipeline_output_seen = False
        self._spcc_seen_in_run = False
        self._spcc_cli_crash_detected = False
        self._recent_process_output.clear()
        self._last_spcc_command = ""
        self._last_output_ts = time.time()

        cmd = build_siril_cli_command(
            siril_cli=siril_cli,
            work_dir=self.work_dir,
            run_ini=run_ini,
            run_ssf=run_ssf,
            offline_mode=not self.network_mode,
        )
        self._append_event(f"开始启动进程（{self._active_mode}），使用 {siril_cli}")
        self._append_event("命令：" + " ".join(cmd))
        proc_env = self._build_env(siril_cli)
        self._append_event(f"Siril 运行时主目录：{proc_env.get('HOME', '')}")
        self._append_event(
            "Siril Python CLI："
            + proc_env.get("SIRIL_PYTHON_CLI", "<未设置>")
        )
        self._append_event(
            f"调试模式: {'ON' if self.debug_mode else 'OFF'}"
        )
        self._append_event(
            f"联网模式: {'ON' if self.network_mode else 'OFF'}"
        )
        self._append_event(f"输入模式: {self.input_mode}")
        self._append_event(
            f"AI 阶段开关: {'ON' if self.ai_stage_enabled else 'OFF'} "
            "(controls stage11)"
        )
        if self._ai_env_sources:
            self._append_event(
                "AI 环境配置来源: " + ", ".join(self._ai_env_sources)
            )
            if self._ai_env_applied_keys:
                self._append_event(
                    "AI 环境已注入键: " + ", ".join(self._ai_env_applied_keys)
                )
        for warning in self._ai_env_warnings:
            self._append_event(f"AI 环境配置警告: {warning}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=proc_env,
            )
        except Exception as e:
            self._append_event(f"启动进程失败：{e}")
            self._run_had_errors = True
            return False, -1

        if self._proc.stdout is None:
            self._append_event("无法捕获进程输出。")
            self._run_had_errors = True
            return False, -1

        out_queue: queue.Queue[str | None] = queue.Queue()
        reader_t = threading.Thread(target=self._reader, args=(self._proc.stdout, out_queue), daemon=True)
        reader_t.start()

        bootstrap_timeout = False
        reader_done = False

        while True:
            drained = False
            while True:
                try:
                    item = out_queue.get_nowait()
                except queue.Empty:
                    break
                drained = True
                if item is None:
                    reader_done = True
                    break
                self._last_output_ts = time.time()
                self._inspect_output_for_errors(item)
                self.log.emit(item)

            proc_ret = self._proc.poll()

            if self._stop_event.is_set() and proc_ret is None:
                self._append_event("已请求停止...")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._append_event("进程未能正常退出，正在强制结束。")
                    self._proc.kill()

            now = time.time()
            if (
                proc_ret is None
                and self._active_mode == "python"
                and self._pyscript_seen_at
                and not self._pipeline_output_seen
                and now - self._pyscript_seen_at > 180
            ):
                bootstrap_timeout = True
                self._run_had_errors = True
                self._python_env_issue = True
                self._append_event(
                    "pyscript 启动超时（>180s）：Siril Python 环境疑似卡住。"
                )
                self._append_event(
                    "提示：关闭 Siril，删除 "
                    f"'{self._siril_venv_dir()}' 后重试。"
                )
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

            proc_ret = self._proc.poll()
            if proc_ret is not None and reader_done and out_queue.empty():
                break

            if not drained:
                time.sleep(0.1)

        # Drain remaining output after process exit.
        while True:
            try:
                item = out_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            self._inspect_output_for_errors(item)
            self.log.emit(item)

        try:
            self._proc.stdout.close()
        except Exception:
            pass
        reader_t.join(timeout=1)

        exit_code = self._proc.returncode if self._proc.returncode is not None else -1
        self._proc = None

        if bootstrap_timeout:
            return False, exit_code

        if exit_code == -11 and self._spcc_seen_in_run:
            self._spcc_cli_crash_detected = True
            self._run_had_errors = True
            self._append_spcc_crash_diagnostics(exit_code)

        if (
            self._active_mode == "python"
            and self._pyscript_seen_at is not None
            and not self._pipeline_output_seen
            and not self._stop_event.is_set()
        ):
            if not self._run_had_errors:
                self._append_event(
                    "pyscript 启动后未检测到流水线阶段输出，本次运行按失败处理。"
                )
            self._run_had_errors = True

        success = exit_code == 0 and not self._run_had_errors and not self._stop_event.is_set()
        return success, exit_code

    def run(self) -> None:
        self.state.emit("Running")
        run_status = "Failed"
        exit_code = -1
        cli_used = ""

        temp_dir = Path(tempfile.mkdtemp(prefix="seestar_superimpose_embedded_"))
        try:
            run_ssf, run_ini, _run_py = self._prepare_runtime_files(temp_dir)

            for attempt, siril_cli in enumerate(self.siril_candidates, start=1):
                if self._stop_event.is_set():
                    run_status = "Stopped"
                    break

                cli_used = str(siril_cli)
                success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                if self._stop_event.is_set():
                    run_status = "Stopped"
                    break

                if success:
                    run_status = "Completed"
                    break

                if self._spcc_cli_crash_detected and not self._spcc_crash_retry_attempted:
                    self._spcc_crash_retry_attempted = True
                    self._force_disable_spcc_for_retry = True
                    self._append_event(
                        "检测到 Siril 在 SPCC 测光阶段崩溃（退出码 -11）。"
                        "正在禁用 SPCC 并重试完整流水线，Stage 4 将改走 PCC/本地校色回退。"
                    )
                    success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                    if self._stop_event.is_set():
                        run_status = "Stopped"
                        break
                    if success:
                        run_status = "Completed"
                        break

                if self._python_env_issue and not self._python_env_repair_attempted:
                    self._python_env_repair_attempted = True
                    self._append_event(
                        "检测到 Siril Python 环境异常。"
                        "正在重置 Siril Python venv，并重试一次..."
                    )
                    if self._reset_siril_python_venv():
                        success, exit_code = self._run_once(siril_cli, run_ssf, run_ini)
                        if self._stop_event.is_set():
                            run_status = "Stopped"
                            break
                        if success:
                            run_status = "Completed"
                            break
                    else:
                        self._append_event(
                            "自动重置 venv 失败。"
                        )

                if attempt < len(self.siril_candidates):
                    self._append_event(
                        "主流水线失败或卡住，"
                        "正在使用备用 Siril 运行时重试完整内置流水线..."
                    )
            else:
                run_status = "Failed"

        except Exception as e:
            self._append_event(f"Worker 内部错误：{e}")
            self._run_had_errors = True
            run_status = "Failed"
            exit_code = -1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if self._stop_event.is_set() and run_status != "Stopped":
            run_status = "Stopped"

        self.done.emit(run_status, exit_code, self._run_had_errors, cli_used)
