#!/usr/bin/env bash
set -euo pipefail

# Build Starun macOS app bundle with:
# - Embedded Siril extracted from packages/siril-1.4.4-arm64-3.dmg
#   (including Siril's CPython 3.12 runtime)
#
# Usage:
#   ./build_macos_app.sh [--app-name NAME] [--output-dir DIR]
#                        [--build-python /path/to/python]
#                        [--gui-entry /path/to/starun_gui_app.py]
#                        [--pipeline-src /path/to/starun.py]
#                        [--config-template /path/to/config.1.4.ini.template]
#                        [--siril-src /path/to/Siril.app]
#                        [--codesign-identity "Developer ID Application: ..."]
#                        [--bundle-profile full|core]
#                        [--offline-resource-pack-dir DIR]
#                        [--help]

APP_NAME="Starun"
BUILD_PYTHON=""
BUILD_PYTHON_OVERRIDE=""
APP_BUNDLE_ID="StarunC"
APP_SHORT_VERSION="0.1"
APP_BUILD_VERSION="1"
MACOS_MIN_VERSION="14.0"
REQUIRED_APP_ARCH="arm64"
CODESIGN_IDENTITY="-"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGES_DIR="$PROJECT_ROOT/packages"
OUTPUT_DIR="$PROJECT_ROOT/release"
GUI_ENTRY="$PROJECT_ROOT/gui/starun_gui_app.py"
APP_LOGO_PNG="$PROJECT_ROOT/gui/Starun.png"
PIPELINE_SRC="$PROJECT_ROOT/pipeline/starun.py"
PIPELINE_REQUIRED_MODULES=(
  stage2_crop_detector.py
  stage3_contract.py
  stage_contracts.py
  task_plan.py
  input_discovery.py
  task_workspace.py
  processing_parameters.py
  final_artifact_identity.py
  presentation_quality.py
  spatial_background_lineage.py
  stage5_handoff.py
  stage8_handoff.py
  stage8_color_rendition.py
  stage8_starless_finish.py
  star_halo_guard.py
  ui_preview.py
)
LOCAL_TEMPLATE="$PROJECT_ROOT/resources/config.1.4.ini.template"
CONFIG_TEMPLATE_IN="$LOCAL_TEMPLATE"
DEFAULT_ENV_SRC="$PROJECT_ROOT/resources/default.env"
SIRIL_PLUGIN_DIR_SRC="$PROJECT_ROOT/resources/siril_plugins"
SIRIL_SPCC_DATABASE_SEED_SRC="$PROJECT_ROOT/resources/siril_spcc_database"
PROJECT_LICENSE_SRC="$PROJECT_ROOT/LICENSE"
PROJECT_NOTICE_SRC="$PROJECT_ROOT/NOTICE"
THIRD_PARTY_NOTICES_SRC="$PROJECT_ROOT/THIRD_PARTY_NOTICES"
APP_REQUIREMENTS="$PROJECT_ROOT/requirements.lock"
GUI_BUILD_REQUIREMENTS="$PROJECT_ROOT/build/requirements-gui-build.lock"
SIRIL_PLUGIN_REQUIREMENTS="$SIRIL_PLUGIN_DIR_SRC/requirements.lock"
SIRIL_PLUGIN_DOWNLOADS_DIR="$SIRIL_PLUGIN_DIR_SRC/downloads"
USER_CONFIG="$HOME/Library/Application Support/org.siril.Siril/siril/config.1.4.ini"

SIRIL_DMG="$PACKAGES_DIR/siril-1.4.4-arm64-3.dmg"
SIRIL_SRC_APP=""
BUNDLE_PROFILE="full"
OFFLINE_RESOURCE_PACK_DIR=""
SIRIL_RUNTIME_STATE="$HOME/Library/Application Support/org.siril.Siril/siril"
SIRIL_SEED_MODULE="$SIRIL_RUNTIME_STATE/.python_module"

# These runtimes are installed into the independent Siril Python 3.12 venv
# from bundled wheels.  Excluding them from the frozen Python 3.13 GUI avoids
# packaging the same scientific stack twice without changing offline runtime
# behavior.
PYINSTALLER_EXCLUDED_MODULES=(
  torch
  torchvision
  onnx
  onnxruntime
  scipy
  pandas
  pyarrow
  numba
  llvmlite
  matplotlib
  botocore
  aiobotocore
  s3fs
  dask
  dask_image
  fsspec
  zarr
  numcodecs
  lz4
  zstandard
  aiohttp
  fastavro
  google_crc32c
  cv2
  sep
  spandrel
  tifffile
  einops
  safetensors
  PyQt6
  setiastrosuitepro
  astroquery
  lightkurve
  photutils
  reproject
  sklearn
  skimage
  networkx
  sympy
  skyfield
  sgp4
  pytest
  _pytest
)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--app-name NAME] [--output-dir DIR]
                   [--build-python PATH]
                   [--gui-entry PATH] [--pipeline-src PATH]
                   [--config-template PATH]
                   [--siril-src /path/to/Siril.app]
                   [--codesign-identity IDENTITY]
                   [--bundle-profile full|core]
                   [--offline-resource-pack-dir DIR]

This script requires the following package files in:
  $PACKAGES_DIR
  - siril-1.4.4-arm64-3.dmg
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-name)
      APP_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --build-python)
      BUILD_PYTHON_OVERRIDE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --gui-entry)
      GUI_ENTRY="$2"
      shift 2
      ;;
    --pipeline-src)
      PIPELINE_SRC="$2"
      shift 2
      ;;
    --config-template)
      CONFIG_TEMPLATE_IN="$2"
      shift 2
      ;;
    --siril-src)
      SIRIL_SRC_APP="$2"
      shift 2
      ;;
    --codesign-identity)
      CODESIGN_IDENTITY="$2"
      shift 2
      ;;
    --bundle-profile)
      BUNDLE_PROFILE="$2"
      shift 2
      ;;
    --offline-resource-pack-dir)
      OFFLINE_RESOURCE_PACK_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Expand explicit "~/" for user provided path arguments.
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"
BUILD_PYTHON_OVERRIDE="${BUILD_PYTHON_OVERRIDE/#\~/$HOME}"
GUI_ENTRY="${GUI_ENTRY/#\~/$HOME}"
PIPELINE_SRC="${PIPELINE_SRC/#\~/$HOME}"
CONFIG_TEMPLATE_IN="${CONFIG_TEMPLATE_IN/#\~/$HOME}"
SIRIL_PLUGIN_DIR_SRC="${SIRIL_PLUGIN_DIR_SRC/#\~/$HOME}"
SIRIL_SRC_APP="${SIRIL_SRC_APP/#\~/$HOME}"
OFFLINE_RESOURCE_PACK_DIR="${OFFLINE_RESOURCE_PACK_DIR/#\~/$HOME}"

case "$BUNDLE_PROFILE" in
  full|core) ;;
  *)
    echo "Unknown bundle profile: $BUNDLE_PROFILE (expected full or core)" >&2
    exit 1
    ;;
esac

if [[ "$BUNDLE_PROFILE" == "core" && -z "$OFFLINE_RESOURCE_PACK_DIR" ]]; then
  OFFLINE_RESOURCE_PACK_DIR="$OUTPUT_DIR/${APP_NAME}-OfflineResources"
fi

die() {
  echo "Error: $*" >&2
  exit 1
}

log() {
  echo "$*"
}

prune_generated_plugin_runtime() {
  local plugin_root="$1"
  local generated_runtime="$plugin_root/cosmic_clarity_runtime"

  case "$plugin_root" in
    "$APP_PATH/Contents/Resources/siril_plugins") ;;
    "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins")
      [[ -n "$OFFLINE_RESOURCE_PACK_DIR" ]] \
        || die "Refusing to prune plugin runtime from an empty resource-pack path"
      ;;
    *)
      die "Refusing to prune generated plugin runtime outside a staging bundle: $plugin_root"
      ;;
  esac

  if [[ -e "$generated_runtime" || -L "$generated_runtime" ]]; then
    rm -rf -- "$generated_runtime"
    log "[BUILD] Excluded generated CosmicClarity runtime cache: $generated_runtime"
  fi
}

verify_bundle_symlinks() {
  local bundle_root="$1"
  local link_path=""
  local link_target=""
  local invalid=0

  while IFS= read -r -d '' link_path; do
    link_target="$(readlink "$link_path")"
    if [[ "$link_target" == /* ]]; then
      echo "[VERIFY] Absolute symlink is not allowed in the app bundle: $link_path -> $link_target" >&2
      invalid=1
      continue
    fi
    if [[ ! -e "$link_path" ]]; then
      echo "[VERIFY] Broken symlink in the app bundle: $link_path -> $link_target" >&2
      invalid=1
    fi
  done < <(find "$bundle_root" -type l -print0)

  [[ "$invalid" -eq 0 ]] || die "App bundle contains invalid symbolic links"
}

remove_old_build_outputs() {
  local app_path="$1"
  local onedir_path="$2"

  if [[ -e "$app_path" ]]; then
    log "[BUILD] Removing existing app bundle: $app_path"
    rm -rf "$app_path"
  fi

  if [[ -e "$onedir_path" ]]; then
    log "[BUILD] Removing stale PyInstaller output: $onedir_path"
    rm -rf "$onedir_path"
  fi
}

resolve_build_python() {
  local py=""
  if [[ -n "$BUILD_PYTHON_OVERRIDE" ]]; then
    py="$BUILD_PYTHON_OVERRIDE"
  elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    py="$SCRIPT_DIR/.venv/bin/python"
  elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    py="$PROJECT_ROOT/.venv/bin/python"
  elif [[ -x "$HOME/.siril/scripts/.venv/bin/python" ]]; then
    py="$HOME/.siril/scripts/.venv/bin/python"
  else
    py="$(command -v python3 || true)"
  fi
  echo "$py"
}

require_exists() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    die "$label not found: $path"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    die "$label is not a file: $path"
  fi
}

embed_project_legal_notices() {
  local destination_root="$1"
  local legal_dir="$destination_root/Legal"

  require_file "$PROJECT_LICENSE_SRC" "Project GPL-3.0-only license"
  require_file "$PROJECT_NOTICE_SRC" "Project copyright notice"
  require_file "$THIRD_PARTY_NOTICES_SRC" "Project third-party notices"

  rm -rf "$legal_dir"
  mkdir -p "$legal_dir"
  cp "$PROJECT_LICENSE_SRC" "$legal_dir/LICENSE"
  cp "$PROJECT_NOTICE_SRC" "$legal_dir/NOTICE"
  cp "$THIRD_PARTY_NOTICES_SRC" "$legal_dir/THIRD_PARTY_NOTICES"
}

verify_locked_plugin_asset() {
  local relative_path="$1"
  local asset_root="${2:-$SIRIL_PLUGIN_DIR_SRC}"
  local absolute_path="$asset_root/$relative_path"
  local checksum_file="$asset_root/asset-checksums.sha256"
  local expected=""
  local actual=""
  require_file "$checksum_file" "[BUILD] plugin asset checksum manifest"
  require_file "$absolute_path" "[BUILD] locked plugin asset"
  expected="$(awk -v path="$relative_path" '$2 == path { print $1; exit }' "$checksum_file")"
  [[ "$expected" =~ ^[0-9a-fA-F]{64}$ ]] \
    || die "[BUILD] missing checksum for plugin asset: $relative_path"
  actual="$(/usr/bin/shasum -a 256 "$absolute_path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] \
    || die "[BUILD] checksum mismatch for plugin asset: $relative_path"
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    die "$label is not a directory: $path"
  fi
}

require_glob_exists() {
  local pattern="$1"
  local label="$2"
  local matches=()
  shopt -s nullglob
  matches=($pattern)
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    die "$label not found: $pattern"
  fi
}

require_any_glob_exists() {
  local label="$1"
  shift
  local pattern=""
  for pattern in "$@"; do
    if compgen -G "$pattern" >/dev/null; then
      return 0
    fi
  done
  die "$label"
}

require_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    die "$label is not executable: $path"
  fi
}

require_apple_silicon_host() {
  local host_arch
  host_arch="$(uname -m)"
  if [[ "$host_arch" != "$REQUIRED_APP_ARCH" ]]; then
    die "[BUILD] Apple Silicon build host required; detected architecture: $host_arch"
  fi
}

macos_wheel_platform_tag() {
  case "$(uname -m)" in
    arm64)
      echo "macosx_14_0_arm64"
      ;;
    *)
      die "[BUILD] Apple Silicon arm64 is required for wheel download: $(uname -m)"
      ;;
  esac
}

configure_app_bundle_metadata() {
  local app_path="$1"
  local info_plist="$app_path/Contents/Info.plist"
  local app_binary="$app_path/Contents/MacOS/$APP_NAME"
  local actual_bundle_id
  local actual_short_version
  local actual_build_version
  local actual_min_version
  local actual_archs

  require_file "$info_plist" "Generated app Info.plist"
  require_executable "$app_binary" "Generated app main binary"

  /usr/bin/plutil -replace CFBundleIdentifier \
    -string "$APP_BUNDLE_ID" "$info_plist"
  /usr/bin/plutil -replace CFBundleShortVersionString \
    -string "$APP_SHORT_VERSION" "$info_plist"
  /usr/bin/plutil -replace CFBundleVersion \
    -string "$APP_BUILD_VERSION" "$info_plist"
  /usr/bin/plutil -replace LSMinimumSystemVersion \
    -string "$MACOS_MIN_VERSION" "$info_plist"
  /usr/bin/plutil -lint "$info_plist" >/dev/null

  actual_bundle_id="$(
    /usr/bin/plutil -extract CFBundleIdentifier raw -o - "$info_plist"
  )"
  actual_short_version="$(
    /usr/bin/plutil -extract CFBundleShortVersionString raw -o - "$info_plist"
  )"
  actual_build_version="$(
    /usr/bin/plutil -extract CFBundleVersion raw -o - "$info_plist"
  )"
  actual_min_version="$(
    /usr/bin/plutil -extract LSMinimumSystemVersion raw -o - "$info_plist"
  )"
  if [[ "$actual_bundle_id" != "$APP_BUNDLE_ID" ]]; then
    die "[BUILD] Unexpected CFBundleIdentifier: $actual_bundle_id"
  fi
  if [[ "$actual_short_version" != "$APP_SHORT_VERSION" ]]; then
    die "[BUILD] Unexpected CFBundleShortVersionString: $actual_short_version"
  fi
  if [[ "$actual_build_version" != "$APP_BUILD_VERSION" ]]; then
    die "[BUILD] Unexpected CFBundleVersion: $actual_build_version"
  fi
  if [[ "$actual_min_version" != "$MACOS_MIN_VERSION" ]]; then
    die "[BUILD] Unexpected LSMinimumSystemVersion: $actual_min_version"
  fi

  actual_archs="$(/usr/bin/lipo -archs "$app_binary")"
  if [[ "$actual_archs" != "$REQUIRED_APP_ARCH" ]]; then
    die "[BUILD] App must be thin arm64; generated architectures: $actual_archs"
  fi

  log "[BUILD] Bundle metadata: id=$APP_BUNDLE_ID, version=$APP_SHORT_VERSION ($APP_BUILD_VERSION)"
  log "[BUILD] Platform requirement: macOS $MACOS_MIN_VERSION or later"
  log "[BUILD] CPU requirement: Apple Silicon ($REQUIRED_APP_ARCH only)"
}

prune_offline_python_wheels() {
  local download_dir="$1"
  local target_abi="cp312"

  "$BUILD_PYTHON" - "$download_dir" "$target_abi" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename

download_dir = Path(sys.argv[1])
target_abi = sys.argv[2]
target_version = int(target_abi.removeprefix("cp"))
allowed_interpreters = {target_abi, "py3"}
allowed_abis = {target_abi, "abi3", "none"}
removed = []
groups = {}

def wheel_preference(tags):
    """Prefer an exact CPython/arm64 wheel over a generic fallback."""
    best = (0, 0, 0)
    for tag in tags:
        if tag.interpreter == target_abi:
            interpreter_score = 3
        elif (
            tag.interpreter.startswith("cp")
            and tag.interpreter[2:].isdigit()
            and int(tag.interpreter[2:]) <= target_version
            and tag.abi == "abi3"
        ):
            interpreter_score = 2
        elif tag.interpreter in {"py3", "py2.py3"}:
            interpreter_score = 1
        else:
            continue
        abi_score = 3 if tag.abi == target_abi else 2 if tag.abi == "abi3" else 1
        platform_score = (
            3 if tag.platform.endswith("_arm64") else
            2 if "universal2" in tag.platform else
            1
        )
        best = max(best, (interpreter_score, abi_score, platform_score))
    return best

for wheel in sorted(download_dir.glob("*.whl")):
    try:
        name, version, _build, tags = parse_wheel_filename(wheel.name)
    except Exception:
        continue
    compatible = any(
        (
            tag.interpreter in allowed_interpreters
            and tag.abi in allowed_abis
        )
        or (
            tag.interpreter.startswith("cp")
            and tag.interpreter[2:].isdigit()
            and int(tag.interpreter[2:]) <= target_version
            and tag.abi == "abi3"
        )
        for tag in tags
    )
    if not compatible:
        wheel.unlink()
        removed.append(wheel)
        continue
    groups.setdefault(canonicalize_name(name), []).append(
        (version, wheel_preference(tags), wheel)
    )

for _name, candidates in sorted(groups.items()):
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    if _name == "setuptools":
        compatible = [item for item in candidates if item[0].major < 82]
        keep = compatible[-1] if compatible else candidates[-1]
    else:
        keep = candidates[-1]
    for _version, _preference, wheel in candidates:
        if wheel == keep[2]:
            continue
        wheel.unlink()
        removed.append(wheel)

print(f"[BUILD] Pruned Python wheels: removed={len(removed)}, kept={len(groups)}")
PY
}

download_offline_python_packages() {
  local platform_tag=""

  require_file "$APP_REQUIREMENTS" "[BUILD] App Python requirements"
  require_file "$SIRIL_PLUGIN_REQUIREMENTS" "[BUILD] Siril plugin Python requirements"
  mkdir -p "$SIRIL_PLUGIN_DOWNLOADS_DIR"

  platform_tag="$(macos_wheel_platform_tag)"
  log "[BUILD] Downloading Python 3.12 offline wheels from: $APP_REQUIREMENTS"
  "$BUILD_PYTHON" -m pip download \
    --require-hashes \
    --only-binary=:all: \
    --python-version 312 \
    --implementation cp \
    --abi cp312 \
    --abi abi3 \
    --platform "$platform_tag" \
    --index-url "https://mirrors.aliyun.com/pypi/simple/" \
    --dest "$SIRIL_PLUGIN_DOWNLOADS_DIR" \
    -r "$APP_REQUIREMENTS"

  log "[BUILD] Downloading Python 3.12 offline wheels from: $SIRIL_PLUGIN_REQUIREMENTS"
  "$BUILD_PYTHON" -m pip download \
    --require-hashes \
    --only-binary=:all: \
    --python-version 312 \
    --implementation cp \
    --abi cp312 \
    --abi abi3 \
    --platform "$platform_tag" \
    --index-url "https://mirrors.aliyun.com/pypi/simple/" \
    --dest "$SIRIL_PLUGIN_DOWNLOADS_DIR" \
    -r "$SIRIL_PLUGIN_REQUIREMENTS"

  prune_offline_python_wheels "$SIRIL_PLUGIN_DOWNLOADS_DIR"
  log "[BUILD] Offline Python package cache updated: $SIRIL_PLUGIN_DOWNLOADS_DIR"
}

resolve_venv_site_packages_dir() {
  local venv_dir="$1"
  local site_dir=""
  site_dir="$(/usr/bin/find "$venv_dir/lib" -maxdepth 2 -type d -name "site-packages" -print -quit 2>/dev/null || true)"
  if [[ -z "$site_dir" ]]; then
    die "[SIRIL] site-packages not found in venv: $venv_dir"
  fi
  echo "$site_dir"
}

repair_siril_seed_site_packages() {
  local site_dir="$1"
  local vendor_dir="$site_dir/pip/_vendor"
  local pkg=""

  # requests/packaging are required by sirilpy but can be missing in some
  # cached venv snapshots; copy pip vendored copies to keep runtime offline.
  if [[ -d "$vendor_dir" ]]; then
    for pkg in packaging requests urllib3 idna certifi; do
      if [[ ! -e "$site_dir/$pkg" && -e "$vendor_dir/$pkg" ]]; then
        /usr/bin/ditto "$vendor_dir/$pkg" "$site_dir/$pkg"
        log "[SIRIL] Seed repair: copied $pkg from pip vendor cache."
      fi
    done
  fi
}

is_minimal_siril_seed_entry() {
  local name="$1"
  case "$name" in
    _distutils_hack|certifi|charset_normalizer|idna|numpy|packaging|pip|pkg_resources|requests|setuptools|sirilpy|urllib3|distutils-precedence.pth|*__mypyc.cpython-312-darwin.so)
      return 0
      ;;
    certifi-*.dist-info|charset_normalizer-*.dist-info|idna-*.dist-info|numpy-*.dist-info|packaging-*.dist-info|pip-*.dist-info|requests-*.dist-info|setuptools-*.dist-info|sirilpy-*.dist-info|sirilpy.egg-info|urllib3-*.dist-info)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

prune_siril_seed_site_packages() {
  local site_dir="$1"
  local before_size=""
  local after_size=""
  local removed_count=0
  local kept_count=0
  local entry=""
  local name=""

  require_dir "$site_dir" "[SIRIL] seed site-packages"

  before_size="$(/usr/bin/du -sh "$site_dir" 2>/dev/null | awk '{print $1}')"
  while IFS= read -r -d '' entry; do
    name="$(basename "$entry")"
    if is_minimal_siril_seed_entry "$name"; then
      kept_count=$((kept_count + 1))
      continue
    fi
    /bin/rm -rf "$entry"
    removed_count=$((removed_count + 1))
  done < <(/usr/bin/find "$site_dir" -mindepth 1 -maxdepth 1 -print0)
  after_size="$(/usr/bin/du -sh "$site_dir" 2>/dev/null | awk '{print $1}')"

  log "[SIRIL] Minimized embedded seed site-packages: ${before_size} -> ${after_size} (kept=${kept_count}, removed=${removed_count})"
}

verify_siril_seed_runtime() {
  local seed_venv="$1"
  local py_bin="$seed_venv/bin/python3.12"
  if [[ ! -x "$py_bin" ]]; then
    log "[SIRIL] Seed python probe skipped (non-executable until runtime rewrite): $py_bin"
    return 0
  fi

  if "$py_bin" - <<'PY'
import importlib
import sys

mods = (
    "sirilpy",
    "numpy",
    "packaging",
    "requests",
    "charset_normalizer",
    "urllib3",
    "idna",
    "certifi",
)
missing = []
versions = {}
for name in mods:
    try:
        mod = importlib.import_module(name)
        versions[name] = getattr(mod, "__version__", "n/a")
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    print("[SIRIL] Offline seed verification failed:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(2)

print(
    "[SIRIL] Offline seed verification OK: "
    + ", ".join(f"{k}={v}" for k, v in versions.items())
)
PY
  then
    return 0
  fi

  log "[SIRIL] Seed python probe warning: runtime import probe failed during build."
  log "[SIRIL] Build continues; runtime will rewrite venv interpreter links to bundled Siril."
}

generate_icns_from_png() {
  local src_png="$1"
  local out_icns="$2"
  local iconset_dir="$BUILD_ROOT/app.iconset"
  local base=""
  local hidpi=""

  require_file "$src_png" "App logo PNG"
  if [[ ! -x "/usr/bin/sips" || ! -x "/usr/bin/iconutil" ]]; then
    die "[BUILD] Missing required macOS tools for icon generation: sips/iconutil"
  fi

  rm -rf "$iconset_dir" "$out_icns"
  mkdir -p "$iconset_dir"

  for base in 16 32 128 256 512; do
    hidpi=$((base * 2))
    /usr/bin/sips -z "$base" "$base" "$src_png" --out "$iconset_dir/icon_${base}x${base}.png" >/dev/null
    /usr/bin/sips -z "$hidpi" "$hidpi" "$src_png" --out "$iconset_dir/icon_${base}x${base}@2x.png" >/dev/null
  done

  /usr/bin/iconutil -c icns "$iconset_dir" -o "$out_icns"
}

has_build_deps() {
  local py_bin="$1"
  "$py_bin" - <<'PY'
import importlib.util, sys
missing = [
    module
    for module in ("PyInstaller", "PySide6", "numpy", "astropy")
    if importlib.util.find_spec(module) is None
]
if missing:
    print(",".join(missing))
    sys.exit(1)
PY
}

report_gui_build_environment() {
  local py_bin="$1"
  "$py_bin" - "${PYINSTALLER_EXCLUDED_MODULES[@]}" <<'PY'
import importlib.util
import sys

present = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is not None]
if present:
    print(
        "[BUILD] Build environment contains runtime-only modules; "
        "PyInstaller exclusions will prevent duplicate collection: "
        + ", ".join(present)
    )
else:
    print("[BUILD] GUI build environment is isolated from runtime-only modules.")
PY
}

verify_pyinstaller_gui_payload() {
  local app_path="$1"
  local analysis_toc="$2"

  require_file "$analysis_toc" "[VERIFY] PyInstaller analysis table"
  "$BUILD_PYTHON" - "$app_path" "$analysis_toc" \
    "${PYINSTALLER_EXCLUDED_MODULES[@]}" <<'PY'
import ast
import sys
from pathlib import Path

app_path = Path(sys.argv[1])
analysis_toc = Path(sys.argv[2])
excluded = tuple(name.casefold() for name in sys.argv[3:])
required = ("pyside6", "numpy", "astropy")


def iter_toc_entries(value):
    if isinstance(value, (list, tuple)):
        if (
            len(value) == 3
            and isinstance(value[0], str)
            and isinstance(value[2], str)
            and value[2] in {
                "BINARY",
                "DATA",
                "EXTENSION",
                "PYMODULE",
                "PYSOURCE",
                "SYMLINK",
            }
        ):
            yield value
            return
        for item in value:
            yield from iter_toc_entries(item)


def is_module(name, module):
    normalized = name.replace("\\", "/").casefold()
    return (
        normalized == module
        or normalized.startswith(module + ".")
        or normalized.startswith(module + "/")
    )


toc = ast.literal_eval(analysis_toc.read_text(encoding="utf-8"))
entry_names = tuple(entry[0] for entry in iter_toc_entries(toc))

missing = [
    module
    for module in required
    if not any(is_module(name, module) for name in entry_names)
]
if missing:
    raise SystemExit(
        "[VERIFY] Frozen GUI is missing required modules: " + ", ".join(missing)
    )

unexpected_modules = sorted(
    {
        module
        for module in excluded
        if any(is_module(name, module) for name in entry_names)
    }
)
if unexpected_modules:
    raise SystemExit(
        "[VERIFY] Runtime-only modules leaked into frozen GUI analysis: "
        + ", ".join(unexpected_modules)
    )

# Catch orphaned native payloads in addition to Python module entries.
native_markers = {
    "torch": ("libtorch", "libc10", "libshm"),
    "pyarrow": ("libarrow", "libparquet"),
}
payload_roots = (
    app_path / "Contents" / "Frameworks",
    app_path / "Contents" / "Resources",
)
unexpected_paths = []
for root in payload_roots:
    if not root.is_dir():
        continue
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix().casefold()
        first = relative.split("/", 1)[0]
        normalized_first = first.replace("-", "_")
        for module in excluded:
            normalized_module = module.replace("-", "_")
            if (
                normalized_first == normalized_module
                or normalized_first.startswith(normalized_module + ".")
                or normalized_first.startswith(normalized_module + "_")
            ):
                unexpected_paths.append(str(path))
                break
            if any(marker in first for marker in native_markers.get(module, ())):
                unexpected_paths.append(str(path))
                break

if unexpected_paths:
    sample = "\n".join(f"  - {path}" for path in unexpected_paths[:12])
    raise SystemExit(
        "[VERIFY] Runtime-only files leaked into frozen GUI payload:\n" + sample
    )

print(
    "[VERIFY] Frozen GUI dependency boundary OK: required="
    + ",".join(required)
    + f" excluded={len(excluded)}"
)
PY
}

detach_mount_safe() {
  local mount_path="$1"
  if [[ -z "$mount_path" ]]; then
    return
  fi
  /usr/bin/hdiutil detach "$mount_path" >/dev/null 2>&1 || \
    /usr/bin/hdiutil detach "$mount_path" -force >/dev/null 2>&1 || true
}

extract_siril_from_dmg() {
  local dmg_path="$1"
  local dst_app="$2"
  local mount_dir="$BUILD_ROOT/siril_mount"
  local raw_img="$BUILD_ROOT/siril_from_bzip2.img"
  local bzip_log="$BUILD_ROOT/siril_bzip2.log"
  local siril_app=""

  rm -rf "$mount_dir" "$raw_img"
  rm -f "$bzip_log"
  mkdir -p "$mount_dir"

  log "[SIRIL] Mounting image: $dmg_path"
  if /usr/bin/hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" "$dmg_path" >/dev/null 2>&1; then
    log "[SIRIL] Direct DMG mount succeeded."
  else
    log "[SIRIL] Direct mount failed. Trying bzip2 fallback..."
    if ! /usr/bin/bzip2 -dc "$dmg_path" > "$raw_img" 2>"$bzip_log"; then
      if [[ ! -s "$raw_img" ]]; then
        [[ -s "$bzip_log" ]] && sed -n '1,60p' "$bzip_log" >&2
        die "[SIRIL] Failed to decompress Siril DMG."
      fi
      log "[SIRIL] bzip2 returned non-zero with trailing data; using generated image."
    fi
    if ! /usr/bin/hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" "$raw_img" >/dev/null 2>&1; then
      if ! /usr/bin/hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" -imagekey diskimage-class=CRawDiskImage "$raw_img" >/dev/null 2>&1; then
        [[ -s "$bzip_log" ]] && sed -n '1,60p' "$bzip_log" >&2
        die "[SIRIL] Failed to mount fallback Siril image."
      fi
    fi
    log "[SIRIL] Fallback mount succeeded."
  fi

  SIRIL_MOUNT_DIR="$mount_dir"
  siril_app="$(/usr/bin/find "$mount_dir" -maxdepth 3 -type d -name "Siril.app" -print -quit || true)"
  if [[ -z "$siril_app" ]]; then
    die "[SIRIL] Siril.app not found in mounted image."
  fi

  rm -rf "$dst_app"
  /usr/bin/ditto "$siril_app" "$dst_app"
  log "[SIRIL] Embedded Siril from: $siril_app"

  detach_mount_safe "$mount_dir"
  SIRIL_MOUNT_DIR=""
}

copy_siril_from_app() {
  local src_app="$1"
  local dst_app="$2"
  require_dir "$src_app" "[SIRIL] Source Siril.app"
  rm -rf "$dst_app"
  /usr/bin/ditto "$src_app" "$dst_app"
  log "[SIRIL] Embedded Siril from app path: $src_app"
}

fix_siril_python_runtime() {
  local siril_app="$1"
  local py_framework="$siril_app/Contents/Frameworks/Python.framework"
  local py_bin="$py_framework/Versions/3.12/bin/python3.12"

  log "[SIRIL] Clearing quarantine/xattrs on embedded Siril.app..."
  /usr/bin/xattr -cr "$siril_app" >/dev/null 2>&1 || true

  if [[ -d "$py_framework" ]]; then
    log "[SIRIL] Re-signing embedded Siril Python.framework runtime files..."
    while IFS= read -r -d '' signed_file; do
      /usr/bin/codesign --force --sign "$CODESIGN_IDENTITY" "$signed_file"
    done < <(/usr/bin/find "$py_framework" -type f \( -perm -111 -o -name "*.dylib" -o -name "*.so" -o -name "Python" \) -print0)

    /usr/bin/codesign --force --deep --sign "$CODESIGN_IDENTITY" "$py_framework"

    if [[ -x "$py_bin" ]]; then
      if ! "$py_bin" -V >/dev/null 2>&1; then
        die "[SIRIL] Embedded Siril Python runtime check failed after re-signing: $py_bin"
      fi
      log "[SIRIL] Embedded Siril Python runtime check passed: $py_bin"
    fi
  fi

  /usr/bin/codesign --force --deep --sign "$CODESIGN_IDENTITY" "$siril_app"
}

embed_siril_offline_python_seed() {
  local app_resources="$1"
  local seed_root="$app_resources/SirilPythonSeed"
  local seed_venv="$BUILD_ROOT/siril_seed_venv"
  local seed_site=""
  local seed_bin=""
  local seed_python=""
  local siril_python="$app_resources/Siril.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
  local rel_py="../../../Siril.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12"

  require_dir "$SIRIL_SEED_MODULE" "Siril offline python seed module"
  require_dir "$SIRIL_PLUGIN_DOWNLOADS_DIR" "Siril offline Python wheels"
  require_executable "$siril_python" "Embedded Siril Python interpreter"

  # The user's Siril runtime venv is mutable and may have been recreated by a
  # different Python version. Build a clean 3.12 seed with the Python runtime
  # that is actually bundled in this app instead of copying that user venv.
  log "[SIRIL] Building clean Python 3.12 offline seed..."
  rm -rf "$seed_venv"
  "$siril_python" -m venv "$seed_venv"
  seed_python="$seed_venv/bin/python3.12"
  require_executable "$seed_python" "Generated Siril offline seed interpreter"
  PIP_CONFIG_FILE=/dev/null "$seed_python" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-index \
    --find-links "$SIRIL_PLUGIN_DOWNLOADS_DIR" \
    --only-binary=:all: \
    setuptools wheel numpy packaging requests
  PIP_CONFIG_FILE=/dev/null "$seed_python" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-index \
    --find-links "$SIRIL_PLUGIN_DOWNLOADS_DIR" \
    --no-build-isolation \
    --no-deps \
    "$SIRIL_SEED_MODULE"

  log "[SIRIL] Embedding clean offline Python seed from: $seed_venv"
  rm -rf "$seed_root"
  mkdir -p "$seed_root"
  /usr/bin/ditto "$seed_venv" "$seed_root/venv"
  /usr/bin/ditto "$SIRIL_SEED_MODULE" "$seed_root/.python_module"

  # Convert absolute venv interpreter links to in-bundle relative links so
  # codesign --strict accepts the bundle.
  seed_bin="$seed_root/venv/bin"
  if [[ -d "$seed_bin" ]]; then
    /bin/rm -f "$seed_bin/python3.12" "$seed_bin/python3" "$seed_bin/python"
    /bin/ln -s "$rel_py" "$seed_bin/python3.12"
    /bin/ln -s "python3.12" "$seed_bin/python3"
    /bin/ln -s "python3.12" "$seed_bin/python"
  fi

  seed_site="$(resolve_venv_site_packages_dir "$seed_root/venv")"
  repair_siril_seed_site_packages "$seed_site"
  prune_siril_seed_site_packages "$seed_site"
  require_exists "$seed_site/sirilpy" "[SIRIL] Seed module sirilpy"
  require_exists "$seed_site/numpy" "[SIRIL] Seed module numpy"
  require_exists "$seed_site/packaging" "[SIRIL] Seed module packaging"
  require_exists "$seed_site/requests" "[SIRIL] Seed module requests"
  require_exists "$seed_site/urllib3" "[SIRIL] Seed module urllib3"
  require_exists "$seed_site/idna" "[SIRIL] Seed module idna"
  require_exists "$seed_site/certifi" "[SIRIL] Seed module certifi"

  verify_siril_seed_runtime "$seed_root/venv"
}

embed_siril_spcc_database_seed() {
  local app_resources="$1"
  local seed_root="$app_resources/SirilSPCCDatabaseSeed"

  require_dir "$SIRIL_SPCC_DATABASE_SEED_SRC" "Siril SPCC database seed source"
  require_file "$SIRIL_SPCC_DATABASE_SEED_SRC/manifest.json" "Siril SPCC seed manifest"
  require_file "$SIRIL_SPCC_DATABASE_SEED_SRC/VERSION.txt" "Siril SPCC seed version list"
  require_file "$SIRIL_SPCC_DATABASE_SEED_SRC/LICENSE.md" "Siril SPCC seed GPLv3 license"
  if ! (
    cd "$SIRIL_SPCC_DATABASE_SEED_SRC"
    /usr/bin/shasum -a 256 -c SHA256SUMS
  ); then
    die "Siril SPCC database seed checksum verification failed"
  fi

  rm -rf "$seed_root"
  mkdir -p "$seed_root"
  cp -R "$SIRIL_SPCC_DATABASE_SEED_SRC/." "$seed_root/"
  log "[SIRIL] Embedded fixed SPCC database seed: $seed_root"
}

require_apple_silicon_host
require_exists "$PROJECT_ROOT" "Project root"
require_exists "$GUI_ENTRY" "GUI entry"
require_file "$APP_LOGO_PNG" "App logo PNG"
require_file "$PIPELINE_SRC" "Pipeline script"
require_file "$PROJECT_LICENSE_SRC" "Project GPL-3.0-only license"
require_file "$PROJECT_NOTICE_SRC" "Project copyright notice"
require_file "$THIRD_PARTY_NOTICES_SRC" "Project third-party notices"
for module_name in "${PIPELINE_REQUIRED_MODULES[@]}"; do
  require_file "$(dirname "$PIPELINE_SRC")/$module_name" "Pipeline runtime module"
done
require_dir "$SIRIL_SPCC_DATABASE_SEED_SRC" "Siril SPCC database seed source"
require_file "$SIRIL_SPCC_DATABASE_SEED_SRC/manifest.json" "Siril SPCC seed manifest"
require_file "$SIRIL_SPCC_DATABASE_SEED_SRC/SHA256SUMS" "Siril SPCC seed checksums"
require_dir "$PACKAGES_DIR" "Packages directory"
require_file "$GUI_BUILD_REQUIREMENTS" "GUI build requirements lock"
if [[ -n "$SIRIL_SRC_APP" ]]; then
  require_dir "$SIRIL_SRC_APP" "Siril app source"
else
  require_file "$SIRIL_DMG" "Siril dmg package"
fi

log "[PATHS] Project root: $PROJECT_ROOT"
log "[PATHS] Packages dir: $PACKAGES_DIR"
if [[ -n "$SIRIL_SRC_APP" ]]; then
  log "[PATHS] Siril app source: $SIRIL_SRC_APP"
else
  log "[PATHS] Siril dmg: $SIRIL_DMG"
fi
BUILD_PYTHON="$(resolve_build_python)"
if [[ -z "$BUILD_PYTHON" ]]; then
  die "Python not found for PyInstaller build."
fi

log "[BUILD] Candidate Python: $BUILD_PYTHON"
log "[BUILD] Checking GUI build dependencies (PyInstaller, PySide6, numpy, astropy)..."
if ! has_build_deps "$BUILD_PYTHON"; then
  if [[ -x "$HOME/.siril/scripts/.venv/bin/python" ]] && has_build_deps "$HOME/.siril/scripts/.venv/bin/python"; then
    BUILD_PYTHON="$HOME/.siril/scripts/.venv/bin/python"
    log "[BUILD] Using fallback Python with dependencies: $BUILD_PYTHON"
  else
    die "Build dependencies missing for $BUILD_PYTHON (requires PyInstaller + PySide6 + numpy + astropy)."
  fi
fi
report_gui_build_environment "$BUILD_PYTHON"

mkdir -p "$OUTPUT_DIR"

# Keep the build and final bundle on the same filesystem. The existing release
# remains untouched until the new bundle has passed every verification step.
BUILD_ROOT="$(mktemp -d "$OUTPUT_DIR/.starun_build.XXXXXX")"
SIRIL_MOUNT_DIR=""
SPEC_DIR="$BUILD_ROOT/spec"
WORK_DIR="$BUILD_ROOT/work"
RES_STAGING="$BUILD_ROOT/resources"
BUILD_DIST_DIR="$BUILD_ROOT/dist"
APP_PATH="$BUILD_DIST_DIR/${APP_NAME}.app"
ONEDIR_PATH="$BUILD_DIST_DIR/$APP_NAME"
FINAL_APP_PATH="$OUTPUT_DIR/${APP_NAME}.app"
APP_ICON_ICNS="$BUILD_ROOT/${APP_NAME}.icns"

cleanup() {
  detach_mount_safe "${SIRIL_MOUNT_DIR:-}"
  rm -rf "$BUILD_ROOT"
}
trap cleanup EXIT

export PYINSTALLER_CONFIG_DIR="$BUILD_ROOT/pyinstaller_config"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

log "[BUILD] Using Python: $BUILD_PYTHON"
log "[BUILD] GUI entry: $GUI_ENTRY"
log "[BUILD] Pipeline script: $PIPELINE_SRC"
log "[BUILD] PyInstaller config dir: $PYINSTALLER_CONFIG_DIR"

PYINSTALLER_EXCLUDE_ARGS=()
for module_name in "${PYINSTALLER_EXCLUDED_MODULES[@]}"; do
  PYINSTALLER_EXCLUDE_ARGS+=(--exclude-module "$module_name")
done
log "[BUILD] Frozen GUI excludes ${#PYINSTALLER_EXCLUDED_MODULES[@]} Siril runtime-only modules."

download_offline_python_packages

log "[BUILD] Cleaning staging output artifacts..."
remove_old_build_outputs "$APP_PATH" "$ONEDIR_PATH"

log "[BUILD] Generating app icon from: $APP_LOGO_PNG"
generate_icns_from_png "$APP_LOGO_PNG" "$APP_ICON_ICNS"

log "[BUILD] Preparing config template..."
mkdir -p "$RES_STAGING"
CONFIG_TEMPLATE="$RES_STAGING/config.1.4.ini.template"
if [[ -f "$CONFIG_TEMPLATE_IN" ]]; then
  cp "$CONFIG_TEMPLATE_IN" "$CONFIG_TEMPLATE"
elif [[ -f "$USER_CONFIG" ]]; then
  cp "$USER_CONFIG" "$CONFIG_TEMPLATE"
else
  cat >"$CONFIG_TEMPLATE" <<'EOCONFIG'
[core]
EOCONFIG
fi

log "[BUILD] Building base app with PyInstaller..."
"$BUILD_PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --osx-bundle-identifier "$APP_BUNDLE_ID" \
  --target-architecture "$REQUIRED_APP_ARCH" \
  --icon "$APP_ICON_ICNS" \
  --name "$APP_NAME" \
  --distpath "$BUILD_DIST_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$SPEC_DIR" \
  "${PYINSTALLER_EXCLUDE_ARGS[@]}" \
  "$GUI_ENTRY"

if [[ ! -d "$APP_PATH" ]]; then
  die "Build failed: app not generated at $APP_PATH"
fi

verify_pyinstaller_gui_payload "$APP_PATH" "$WORK_DIR/$APP_NAME/Analysis-00.toc"
configure_app_bundle_metadata "$APP_PATH"

# PyInstaller may leave a sibling onedir folder; remove it to avoid confusion.
if [[ -d "$ONEDIR_PATH" ]]; then
  log "[BUILD] Removing PyInstaller onedir output: $ONEDIR_PATH"
  rm -rf "$ONEDIR_PATH"
fi

APP_RESOURCES="$APP_PATH/Contents/Resources"
APP_FRAMEWORKS="$APP_PATH/Contents/Frameworks"
mkdir -p "$APP_RESOURCES/pipeline"
mkdir -p "$APP_FRAMEWORKS"
embed_project_legal_notices "$APP_RESOURCES"

log "[BUILD] Embedding core pipeline/config resources..."
cp "$CONFIG_TEMPLATE" "$APP_RESOURCES/config.1.4.ini.template"
PIPELINE_SRC_DIR="$(cd "$(dirname "$PIPELINE_SRC")" && pwd)"
find "$APP_RESOURCES/pipeline" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
find "$PIPELINE_SRC_DIR" -maxdepth 1 -type f -name "*.py" -exec cp {} "$APP_RESOURCES/pipeline/" \;
if [[ -d "$PIPELINE_SRC_DIR/stages" ]]; then
  cp -R "$PIPELINE_SRC_DIR/stages" "$APP_RESOURCES/pipeline/stages"
fi
if [[ -d "$PIPELINE_SRC_DIR/configs" ]]; then
  cp -R "$PIPELINE_SRC_DIR/configs" "$APP_RESOURCES/pipeline/configs"
fi
if [[ ! -f "$APP_RESOURCES/pipeline/starun.py" ]]; then
  cp "$PIPELINE_SRC" "$APP_RESOURCES/pipeline/starun.py"
fi
require_file "$APP_RESOURCES/config.1.4.ini.template" "[VERIFY] Embedded config template"
require_file "$APP_RESOURCES/pipeline/starun.py" "[VERIFY] Embedded pipeline script"
for module_name in "${PIPELINE_REQUIRED_MODULES[@]}"; do
  require_file "$APP_RESOURCES/pipeline/$module_name" "[VERIFY] Embedded pipeline runtime module"
done

log "[SIRIL] Embedding Siril..."
if [[ -n "$SIRIL_SRC_APP" ]]; then
  copy_siril_from_app "$SIRIL_SRC_APP" "$APP_RESOURCES/Siril.app"
else
  extract_siril_from_dmg "$SIRIL_DMG" "$APP_RESOURCES/Siril.app"
fi
fix_siril_python_runtime "$APP_RESOURCES/Siril.app"
embed_siril_offline_python_seed "$APP_RESOURCES"
embed_siril_spcc_database_seed "$APP_RESOURCES"

if [[ -f "$DEFAULT_ENV_SRC" ]]; then
  cp "$DEFAULT_ENV_SRC" "$APP_RESOURCES/default.env"
  log "[BUILD] Embedded default env file: $DEFAULT_ENV_SRC"
fi
if [[ -d "$SIRIL_PLUGIN_DIR_SRC" ]]; then
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/setiastrosuitepro-*.whl" "[BUILD] setiastrosuitepro wheel (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/appdirs-*.whl" "[BUILD] appdirs wheel (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/ml_dtypes-*.whl" "[BUILD] ml-dtypes wheel (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/onnx-*.whl" "[BUILD] onnx wheel (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/onnxruntime-*.whl" "[BUILD] onnxruntime wheel (run resources/siril_plugins/download_siril_plugins.sh)"
  require_any_glob_exists \
    "[BUILD] PyQt6 wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/pyqt6-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/PyQt6-*.whl"
  require_any_glob_exists \
    "[BUILD] PyQt6_Qt6 wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/pyqt6_qt6-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/PyQt6_Qt6-*.whl"
  require_any_glob_exists \
    "[BUILD] pyqt6_sip wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/pyqt6_sip-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/PyQt6_sip-*.whl"
  require_any_glob_exists \
    "[BUILD] PySide6 wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/pyside6-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/PySide6-*.whl"
  require_any_glob_exists \
    "[BUILD] PySide6_Addons wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/pyside6_addons-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/PySide6_Addons-*.whl"
  require_any_glob_exists \
    "[BUILD] PySide6_Essentials wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/pyside6_essentials-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/PySide6_Essentials-*.whl"
  require_any_glob_exists \
    "[BUILD] shiboken6 wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/shiboken6-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/Shiboken6-*.whl"
  require_any_glob_exists \
    "[BUILD] astropy wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/astropy-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/Astropy-*.whl"
  require_any_glob_exists \
    "[BUILD] scipy wheel missing (run resources/siril_plugins/download_siril_plugins.sh)" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/scipy-*.whl" \
    "$SIRIL_PLUGIN_DIR_SRC/downloads/Scipy-*.whl"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/tifffile-*.whl" "[BUILD] tifffile wheel (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/sep-*.whl" "[BUILD] sep wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/spandrel-*.whl" "[BUILD] spandrel wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/lz4-*.whl" "[BUILD] lz4 wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/zstandard-*.whl" "[BUILD] zstandard wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/exifread-*.whl" "[BUILD] exifread wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/opencv_python_headless-*.whl" "[BUILD] opencv-python-headless wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/requests-*.whl" "[BUILD] requests wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/urllib3-*.whl" "[BUILD] urllib3 wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/idna-*.whl" "[BUILD] idna wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/certifi-*.whl" "[BUILD] certifi wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/charset_normalizer-*.whl" "[BUILD] charset_normalizer wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/setuptools-*.whl" "[BUILD] setuptools wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/wheel-*.whl" "[BUILD] wheel package missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/einops-*.whl" "[BUILD] einops wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/safetensors-*.whl" "[BUILD] safetensors wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/torch-*.whl" "[BUILD] torch wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  require_glob_exists "$SIRIL_PLUGIN_DIR_SRC/downloads/torchvision-*.whl" "[BUILD] torchvision wheel missing (run resources/siril_plugins/download_siril_plugins.sh)"
  if [[ ! -f "$SIRIL_PLUGIN_DIR_SRC/vendor/siril-scripts/SyQon/Starless.py" ]] \
    && [[ ! -f "$SIRIL_PLUGIN_DIR_SRC/vendor/siril-scripts/siril-scripts/SyQon/Starless.py" ]]; then
    die "[BUILD] SyQon Starless script missing (run resources/siril_plugins/download_siril_plugins.sh)"
  fi
  require_file "$SIRIL_PLUGIN_DIR_SRC/syqon_starless/zenith.pt" "[BUILD] SyQon Zenith model"
  require_file "$SIRIL_PLUGIN_DIR_SRC/syqon_starless/zenith.pt.sha256" "[BUILD] SyQon Zenith checksum"
  verify_locked_plugin_asset "patches/apply_syqon_offline_model_patch.py"
  verify_locked_plugin_asset "patches/apply_graxpert_ai_runtime_patch.py"
  verify_locked_plugin_asset "vendor/siril-scripts/SyQon/Starless.py"
  verify_locked_plugin_asset "vendor/siril-scripts/processing/GraXpert-AI.py"
  verify_locked_plugin_asset "syqon_starless/zenith.pt"
  require_file "$SIRIL_PLUGIN_DIR_SRC/graxpert/deconvolution-object-ai-models/1.0.1/model.onnx" "[BUILD] GraXpert object deconvolution model"
  require_file "$SIRIL_PLUGIN_DIR_SRC/graxpert/bge-ai-models/model_v2_0_1/model.onnx" "[BUILD] GraXpert BGE model"
  verify_locked_plugin_asset "graxpert/bge-ai-models/model_v2_0_1/model.onnx"
  require_file "$SIRIL_PLUGIN_DIR_SRC/cosmic_clarity/deep_denoise_mono_AI4.pth" "[BUILD] CosmicClarity mono denoise model"
  require_file "$SIRIL_PLUGIN_DIR_SRC/cosmic_clarity/deep_denoise_color_AI4.pth" "[BUILD] CosmicClarity color denoise model"
  require_file "$SIRIL_PLUGIN_DIR_SRC/cosmic_clarity/deep_sharp_stellar_AI4.pth" "[BUILD] CosmicClarity stellar sharpen model"
  require_file "$SIRIL_PLUGIN_DIR_SRC/cosmic_clarity/deep_nonstellar_sharp_conditional_psf_AI4.pth" "[BUILD] CosmicClarity nonstellar sharpen model"
  require_executable "$SIRIL_PLUGIN_DIR_SRC/bin/CosmicClarity" "[BUILD] CosmicClarity classic wrapper"
  rm -rf "$APP_RESOURCES/siril_plugins"
  if [[ "$BUNDLE_PROFILE" == "full" ]]; then
    ditto "$SIRIL_PLUGIN_DIR_SRC" "$APP_RESOURCES/siril_plugins"
    prune_generated_plugin_runtime "$APP_RESOURCES/siril_plugins"
    "$BUILD_PYTHON" "$APP_RESOURCES/siril_plugins/patches/apply_graxpert_ai_runtime_patch.py" \
      "$APP_RESOURCES/siril_plugins"
    verify_locked_plugin_asset \
      "vendor/siril-scripts/processing/GraXpert-AI.py" \
      "$APP_RESOURCES/siril_plugins"
    "$BUILD_PYTHON" "$APP_RESOURCES/siril_plugins/patches/apply_syqon_offline_model_patch.py" \
      "$APP_RESOURCES/siril_plugins"
    log "[BUILD] Full Offline profile: plugin wheels and models are embedded in the app."
  else
    [[ -n "$OFFLINE_RESOURCE_PACK_DIR" ]] || die "Core profile resource pack path is empty"
    [[ "$OFFLINE_RESOURCE_PACK_DIR" != "/" ]] || die "Refusing to use / as the resource pack path"
    rm -rf "$OFFLINE_RESOURCE_PACK_DIR"
    mkdir -p "$OFFLINE_RESOURCE_PACK_DIR"
    embed_project_legal_notices "$OFFLINE_RESOURCE_PACK_DIR"
    ditto "$SIRIL_PLUGIN_DIR_SRC" "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins"
    prune_generated_plugin_runtime "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins"
    "$BUILD_PYTHON" \
      "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/patches/apply_graxpert_ai_runtime_patch.py" \
      "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins"
    verify_locked_plugin_asset \
      "vendor/siril-scripts/processing/GraXpert-AI.py" \
      "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins"
    "$BUILD_PYTHON" \
      "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/patches/apply_syqon_offline_model_patch.py" \
      "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins"
    log "[BUILD] Core profile: app excludes offline plugin wheels and models."
    log "[BUILD] Offline resource pack: $OFFLINE_RESOURCE_PACK_DIR"
  fi
  log "[BUILD] Runtime plugin wheels will be installed lazily from the offline cache on first app run."
else
  log "[BUILD] Siril plugin dir not found, skip embedding: $SIRIL_PLUGIN_DIR_SRC"
fi

chmod +x "$APP_RESOURCES/Siril.app/Contents/MacOS/siril-cli" || true

require_exists "$APP_RESOURCES/Siril.app" "[VERIFY] Embedded Siril"
require_exists "$APP_RESOURCES/SirilPythonSeed/venv/bin/python3.12" "[VERIFY] Embedded Siril offline venv seed"
require_exists "$APP_RESOURCES/SirilPythonSeed/.python_module/sirilpy" "[VERIFY] Embedded Siril offline module seed"
require_file "$APP_RESOURCES/SirilSPCCDatabaseSeed/manifest.json" "[VERIFY] Embedded Siril SPCC database seed"
require_file "$APP_RESOURCES/Legal/LICENSE" "[VERIFY] Embedded project GPL-3.0-only license"
require_file "$APP_RESOURCES/Legal/NOTICE" "[VERIFY] Embedded project copyright notice"
require_file "$APP_RESOURCES/Legal/THIRD_PARTY_NOTICES" "[VERIFY] Embedded third-party notices"
GAIA_CATALOG_SCAN_ROOTS=("$APP_PATH")
if [[ "$BUNDLE_PROFILE" == "core" ]]; then
  GAIA_CATALOG_SCAN_ROOTS+=("$OFFLINE_RESOURCE_PACK_DIR")
fi
if /usr/bin/find "${GAIA_CATALOG_SCAN_ROOTS[@]}" -type f \
  \( -name "siril_cat*_healpix*_astro*.dat" \
     -o -name "siril_cat*_healpix*_xpsamp*.dat" \
     -o -name "gaia_astrometric.dat" \
     -o -name "gaia_photometric.dat" \) \
  -print -quit | /usr/bin/grep -q .; then
  die "[VERIFY] Gaia catalogues must be downloaded to runtime home, never bundled in the app"
fi
if [[ "$BUNDLE_PROFILE" == "full" ]]; then
  require_dir "$APP_RESOURCES/siril_plugins/downloads" "[VERIFY] Embedded offline wheels"
  require_file "$APP_RESOURCES/siril_plugins/syqon_starless/zenith.pt" "[VERIFY] Embedded SyQon model"
  require_file "$APP_RESOURCES/siril_plugins/syqon_starless/zenith.pt.sha256" "[VERIFY] Embedded SyQon checksum"
  require_file "$APP_RESOURCES/siril_plugins/graxpert/deconvolution-object-ai-models/1.0.1/model.onnx" "[VERIFY] Embedded GraXpert object deconvolution model"
  require_dir "$APP_RESOURCES/siril_plugins/cosmic_clarity" "[VERIFY] Embedded CosmicClarity models"
else
  require_dir "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/downloads" "[VERIFY] Core offline resource wheels"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/Legal/LICENSE" "[VERIFY] Core offline resource GPL-3.0-only license"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/Legal/NOTICE" "[VERIFY] Core offline resource copyright notice"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/Legal/THIRD_PARTY_NOTICES" "[VERIFY] Core offline resource third-party notices"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/syqon_starless/zenith.pt" "[VERIFY] Core offline resource SyQon model"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/syqon_starless/zenith.pt.sha256" "[VERIFY] Core offline resource SyQon checksum"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/graxpert/deconvolution-object-ai-models/1.0.1/model.onnx" "[VERIFY] Core offline resource GraXpert object deconvolution model"
  require_file "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/graxpert/bge-ai-models/model_v2_0_1/model.onnx" "[VERIFY] Core offline resource GraXpert BGE model"
  require_dir "$OFFLINE_RESOURCE_PACK_DIR/siril_plugins/cosmic_clarity" "[VERIFY] Core offline resource CosmicClarity models"
fi

log "[BUILD] Applying deep signing with identity: $CODESIGN_IDENTITY"
/usr/bin/xattr -cr "$APP_PATH" >/dev/null 2>&1 || true
verify_bundle_symlinks "$APP_PATH"
codesign --force --deep --sign "$CODESIGN_IDENTITY" "$APP_PATH"
codesign --verify --deep --strict "$APP_PATH"

log "[BUILD] Publishing verified app bundle..."
PREVIOUS_APP_PATH="$OUTPUT_DIR/.${APP_NAME}.previous.$$"
if [[ -e "$PREVIOUS_APP_PATH" ]]; then
  die "Refusing to overwrite temporary publish backup: $PREVIOUS_APP_PATH"
fi
if [[ -e "$FINAL_APP_PATH" ]]; then
  /bin/mv "$FINAL_APP_PATH" "$PREVIOUS_APP_PATH"
fi
if ! /bin/mv "$APP_PATH" "$FINAL_APP_PATH"; then
  if [[ -e "$PREVIOUS_APP_PATH" && ! -e "$FINAL_APP_PATH" ]]; then
    /bin/mv "$PREVIOUS_APP_PATH" "$FINAL_APP_PATH"
  fi
  die "Failed to publish verified app bundle: $FINAL_APP_PATH"
fi
if ! codesign --verify --deep --strict "$FINAL_APP_PATH"; then
  rm -rf "$FINAL_APP_PATH"
  if [[ -e "$PREVIOUS_APP_PATH" ]]; then
    /bin/mv "$PREVIOUS_APP_PATH" "$FINAL_APP_PATH"
  fi
  die "Published app signature verification failed: $FINAL_APP_PATH"
fi
if [[ -e "$PREVIOUS_APP_PATH" ]]; then
  rm -rf "$PREVIOUS_APP_PATH"
fi
APP_PATH="$FINAL_APP_PATH"

echo
log "Build complete."
log "App path: $APP_PATH"
if [[ "$BUNDLE_PROFILE" == "core" ]]; then
  log "Offline resource pack: $OFFLINE_RESOURCE_PACK_DIR"
fi
