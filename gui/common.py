#!/usr/bin/env python3
"""Shared runtime configuration and helpers for the Seestar GUI."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from math import floor
from pathlib import Path


__all__ = [
    "AI_ENV_ALLOWED_KEYS",
    "AI_ENV_OVERRIDE_NAME",
    "AI_ENV_RESOURCE_REL",
    "APP_RUNTIME_HOME_REL",
    "COSMIC_CLARITY_BUNDLE_REL",
    "COSMIC_CLARITY_REQUIRED_MODEL_FILES",
    "DEFAULT_ENV_RESOURCE_REL",
    "DEFAULT_SIRIL_CONFIG_TEMPLATE",
    "INPUT_MODE_AUTO",
    "INPUT_MODE_LINEAR_RESUME",
    "INPUT_MODE_STAGE2_CORRECTED_RESUME",
    "LINEAR_RESUME_INPUT_NAME",
    "PIPELINE_EXCLUDE_PREFIXES",
    "PIPELINE_EXCLUDE_SUBSTRINGS",
    "PIPELINE_EXCLUDE_SUFFIXES",
    "PIPELINE_RESOURCE_REL",
    "SIRIL_COSMIC_REQUIRED_WHEEL_LABELS",
    "SIRIL_PLUGIN_RESOURCE_REL",
    "SIRIL_REQUIRED_SITE_PACKAGES",
    "SIRIL_STARLESS_REQUIRED_WHEEL_LABELS",
    "SIRIL_VENDOR_FALLBACK_PACKAGES",
    "STAGE2_CORRECTED_INPUT_NAME",
    "SYQON_STARLESS_BUNDLE_REL",
    "apply_siril_runtime_patches",
    "build_siril_cli_command",
    "compute_siril_cpu_limit",
    "default_pipeline_path",
    "default_runtime_home",
    "default_siril_plugin_dir",
    "is_frozen",
    "normalize_siril_config_template",
    "parse_ai_env_file",
    "project_root",
    "repair_site_packages_from_pip_vendor",
    "resolve_existing_path",
    "resolve_siril_scripts_root",
    "resolve_venv_site_packages",
    "resource_root",
    "scrub_python_env",
    "shell_quote_path",
    "siril_state_root_from_home",
    "verify_siril_offline_seed_venv",
]


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
catalogue_gaia_astro=
catalogue_gaia_photo=
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
auto_update_scripts=false
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


def normalize_siril_config_template(
    config_text: str,
    *,
    gaia_photo_catalog: Path | None = None,
    gaia_astro_catalog: Path | None = None,
) -> str:
    removed_keys = {"starnet_exe", "starnet_weights"}
    lines = []
    for raw_line in config_text.splitlines():
        key = raw_line.split("=", 1)[0].strip().lower() if "=" in raw_line else ""
        if key in removed_keys:
            continue
        if key == "catalogue_gaia_astro" and gaia_astro_catalog is not None:
            lines.append(f"catalogue_gaia_astro={gaia_astro_catalog.expanduser()}")
            continue
        if key == "catalogue_gaia_photo":
            if gaia_photo_catalog is not None:
                lines.append(f"catalogue_gaia_photo={gaia_photo_catalog.expanduser()}")
            else:
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
        return normalize_siril_config_template(
            DEFAULT_SIRIL_CONFIG_TEMPLATE,
            gaia_photo_catalog=gaia_photo_catalog,
            gaia_astro_catalog=gaia_astro_catalog,
        )
    if not any(line.strip() == "[core]" for line in lines):
        lines.extend(["", "[core]"])

    existing_keys = {
        line.split("=", 1)[0].strip().lower()
        for line in lines
        if "=" in line
    }
    missing_catalog_lines = []
    if gaia_astro_catalog is not None and "catalogue_gaia_astro" not in existing_keys:
        missing_catalog_lines.append(
            f"catalogue_gaia_astro={gaia_astro_catalog.expanduser()}"
        )
    if gaia_photo_catalog is not None and "catalogue_gaia_photo" not in existing_keys:
        missing_catalog_lines.append(
            f"catalogue_gaia_photo={gaia_photo_catalog.expanduser()}"
        )
    if missing_catalog_lines:
        core_index = next(
            index for index, line in enumerate(lines) if line.strip() == "[core]"
        )
        lines[core_index + 1 : core_index + 1] = missing_catalog_lines

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
        "SEESTAR_AI_ADVISOR_MODE",
        "SEESTAR_AI_STAGE6_ENABLE",
        "SEESTAR_AI_STAGE7_ENABLE",
        "SEESTAR_AI_STAGE8_ENABLE",
        "SEESTAR_AI_ARTISTIC_DERIVATIVE_ENABLED",
        "SEESTAR_AI_ARTISTIC_ENDPOINT",
        "SEESTAR_AI_ARTISTIC_MODEL",
        "SEESTAR_AI_ARTISTIC_API_KEY",
        "SEESTAR_AI_ARTISTIC_PROMPT",
        "SEESTAR_AI_ARTISTIC_TIMEOUT_SEC",
        "SEESTAR_OUTPUT_FORMAT",
        "SEESTAR_FORCE_REVIEW_ONLY_OUTPUT",
        "SEESTAR_DENOISE_ENABLE",
        "SEESTAR_DENOISE_FORCE",
        "SEESTAR_SYQON_GPU",
        "SEESTAR_SYQON_TIMEOUT_SEC",
        "SEESTAR_BOOTSTRAP_TIMEOUT_SEC",
        "SEESTAR_WATCHDOG_IDLE_TIMEOUT_SEC",
        "SEESTAR_EXPORT_TAIL_TIMEOUT_SEC",
        "SEESTAR_TEMP_CLEANUP_TIMEOUT_SEC",
        "SEESTAR_SIRILPY_TIMEOUT_SEC",
        "SEESTAR_WORKFLOW_PLUGIN_PROBE",
        "SEESTAR_STAGE4_PLATESOLVE_ENABLE",
        "SEESTAR_STAGE4_PLATESOLVE_CATALOGS",
        "SEESTAR_STAGE4_FILTER_HINT",
        "SEESTAR_STAGE4_PCC_TIMEOUT_SEC",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_ENABLE",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_MIN_PIXELS",
        "SEESTAR_STAGE4_LOCAL_STAR_WB_GAIN_LIMIT",
        "SEESTAR_STAGE4_LOCAL_STAR_MASK_RADIUS",
        "SEESTAR_STAGE4_LOCAL_STAR_MASK_COVERAGE_MAX",
        "SEESTAR_ABERRATION_API_ENABLE",
        "SEESTAR_ABERRATION_PROVIDER",
        "SEESTAR_OPTIONAL_COLOR_TRANSFORM",
        "SEESTAR_COSMIC_CLASSIC_ENABLE",
        "SEESTAR_COSMIC_CLARITY_EXECUTABLE",
        "SEESTAR_COSMIC_CLASSIC_GPU",
        "SEESTAR_COSMIC_NATIVE_GPU",
        "SEESTAR_STAGE5_BUILTIN_DENOISE_MOD",
        "SEESTAR_STAGE5_DECONV_ENABLE",
        "SEESTAR_STAGE5_GRAXPERT_DECONV_ENABLE",
        "SEESTAR_STAGE5_RL_MAXSTARS",
        "SEESTAR_STAGE5_RL_PSF_KS",
        "SEESTAR_STAGE5_RL_ITERS",
        "SEESTAR_STAGE5_RL_ALPHA",
        "SEESTAR_STAGE5_RL_GDSTEP",
        "SEESTAR_STAGE5_RL_STOP",
        "SEESTAR_STAGE5_GRAXPERT_DECONV_STRENGTH",
        "SEESTAR_GRAXPERT_OBJECT_MODEL_PATH",
        "SEESTAR_STAGE7_QUALITY_RETRY_MAX",
        "SEESTAR_STAGE7_SKIP_UNREADY_STARLESS",
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
        env_override = os.environ.get("SEESTAR_OFFLINE_RESOURCE_ROOT", "").strip()
        env_candidates: list[Path] = []
        if env_override:
            override = Path(env_override).expanduser()
            env_candidates.extend((override / "siril_plugins", override))

        app_name = resources.parent.parent.stem
        distribution_root = resources.parent.parent.parent
        external_candidates = [
            distribution_root / f"{app_name}-OfflineResources" / "siril_plugins",
            distribution_root / "SeestarSuperimpose-OfflineResources" / "siril_plugins",
            Path.home()
            / "Library/Application Support/SeestarSuperimpose/offline_resources/siril_plugins",
        ]
        embedded = resources / "siril_plugins"
        candidates = [*env_candidates, embedded, *external_candidates]
        return next((path for path in candidates if path.is_dir()), embedded)
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
        # The frozen GUI uses PySide6, while Siril scripts use their own
        # PyQt6/PySide6 wheels.  Inheriting the GUI's Qt plugin paths makes a
        # PyQt6 script load the app-bundled PySide6 Qt frameworks as well,
        # which aborts Cocoa/offscreen plugin initialization on macOS.
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_QPA_PLATFORM",
        "QML2_IMPORT_PATH",
        "QML_IMPORT_PATH",
        "QTWEBENGINEPROCESS_PATH",
        "QTWEBENGINE_RESOURCES_PATH",
        "QTWEBENGINE_LOCALES_PATH",
        "QT_API",
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
