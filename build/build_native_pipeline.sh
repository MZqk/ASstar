#!/usr/bin/env bash
set -euo pipefail

# Build selected Starun pipeline modules as CPython 3.12 macOS arm64
# extension bundles.  This is a technical PoC artifact builder; it does not
# mutate release/Starun.app or claim public-release eligibility.

APP_NAME="Starun"
MACOS_MIN_VERSION="14.0"
EXPECTED_CYTHON_VERSION="3.3.0"
CODESIGN_IDENTITY="-"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_SRC_DIR="$PROJECT_ROOT/pipeline"
OUTPUT_DIR="$PROJECT_ROOT/dist/native-pipeline-cp312-arm64"
BUILD_PYTHON=""
TARGET_PYTHON=""
SOURCE_ARCHIVE=""
SOURCE_ARCHIVE_SHA256=""
PRINT_MODULES=0

NATIVE_MODULES=(
  stage3_contract
  background_sampling
  stage4_auto_reference
  local_adjustments
  stage9_quality
)

usage() {
  printf '%s\n' \
    "Usage:" \
    "  $(basename "$0") [options]" \
    "" \
    "Options:" \
    "  --build-python PATH          Python containing Cython $EXPECTED_CYTHON_VERSION" \
    "  --target-python PATH         Siril CPython 3.12 arm64 interpreter" \
    "  --pipeline-src-dir DIR       Pipeline source directory" \
    "  --output-dir DIR             Output root (default: dist/native-pipeline-cp312-arm64)" \
    "  --source-archive PATH        Optional exact source archive to hash" \
    "  --codesign-identity VALUE    Native module signing identity (default: ad-hoc '-')" \
    "  --print-modules              Print selected import names and exit" \
    "  --help                       Show this help"
}

die() {
  echo "[NATIVE][ERROR] $*" >&2
  exit 1
}

log() {
  echo "[NATIVE] $*"
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || die "$label is not a file: $path"
}

require_executable() {
  local path="$1"
  local label="$2"
  [[ -x "$path" ]] || die "$label is not executable: $path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-python)
      [[ $# -ge 2 ]] || die "--build-python requires a value"
      BUILD_PYTHON="$2"
      shift 2
      ;;
    --target-python)
      [[ $# -ge 2 ]] || die "--target-python requires a value"
      TARGET_PYTHON="$2"
      shift 2
      ;;
    --pipeline-src-dir)
      [[ $# -ge 2 ]] || die "--pipeline-src-dir requires a value"
      PIPELINE_SRC_DIR="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --source-archive)
      [[ $# -ge 2 ]] || die "--source-archive requires a value"
      SOURCE_ARCHIVE="$2"
      shift 2
      ;;
    --codesign-identity)
      [[ $# -ge 2 ]] || die "--codesign-identity requires a value"
      CODESIGN_IDENTITY="$2"
      shift 2
      ;;
    --print-modules)
      PRINT_MODULES=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

if [[ "$PRINT_MODULES" == "1" ]]; then
  printf '%s\n' "${NATIVE_MODULES[@]}"
  exit 0
fi

if [[ -z "$BUILD_PYTHON" ]]; then
  if [[ -x "$SCRIPT_DIR/native/venv/bin/python3" ]]; then
    BUILD_PYTHON="$SCRIPT_DIR/native/venv/bin/python3"
  elif [[ -x "$SCRIPT_DIR/.venv/bin/python3" ]]; then
    BUILD_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
  else
    BUILD_PYTHON="$(command -v python3 || true)"
  fi
fi

if [[ -z "$TARGET_PYTHON" ]]; then
  seed_python="$PROJECT_ROOT/release/$APP_NAME.app/Contents/Resources/SirilPythonSeed/venv/bin/python3.12"
  bundled_python="$PROJECT_ROOT/release/$APP_NAME.app/Contents/Resources/Siril.app/Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
  if [[ -x "$seed_python" ]]; then
    TARGET_PYTHON="$seed_python"
  elif [[ -x "$bundled_python" ]]; then
    TARGET_PYTHON="$bundled_python"
  fi
fi

require_executable "$BUILD_PYTHON" "Build Python"
require_executable "$TARGET_PYTHON" "Target CPython"
require_file "$SCRIPT_DIR/verify_native_pipeline.py" "Native verifier"
require_file "$SCRIPT_DIR/requirements-native-pipeline.lock" "Native build lock"
[[ -d "$PIPELINE_SRC_DIR" ]] || die "Pipeline source directory not found: $PIPELINE_SRC_DIR"
PIPELINE_SRC_DIR="$(cd "$PIPELINE_SRC_DIR" && pwd)"
if [[ "$PIPELINE_SRC_DIR" != "$PROJECT_ROOT/pipeline" ]]; then
  die "PoC source must resolve to the current repository pipeline: $PROJECT_ROOT/pipeline"
fi
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon arm64 build host required"
require_executable "/usr/bin/clang" "Apple clang"
require_executable "/usr/bin/lipo" "lipo"
require_executable "/usr/bin/otool" "otool"
require_executable "/usr/bin/codesign" "codesign"

if [[ -n "$SOURCE_ARCHIVE" ]]; then
  require_file "$SOURCE_ARCHIVE" "Source archive"
  SOURCE_ARCHIVE_SHA256="$(shasum -a 256 "$SOURCE_ARCHIVE" | awk '{print $1}')"
fi

build_metadata="$("$BUILD_PYTHON" -c '
import platform
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\t{platform.machine()}")
')"
IFS=$'\t' read -r build_python_version build_python_arch <<<"$build_metadata"
[[ "$build_python_version" == 3.13.* ]] || die "Native build Python must be 3.13; found $build_python_version"
[[ "$build_python_arch" == "arm64" ]] || die "Native build Python must be arm64; found $build_python_arch"

cython_version="$("$BUILD_PYTHON" -c 'import Cython; print(Cython.__version__)' 2>/dev/null || true)"
[[ "$cython_version" == "$EXPECTED_CYTHON_VERSION" ]] || die \
  "Cython $EXPECTED_CYTHON_VERSION required in $BUILD_PYTHON; found: ${cython_version:-missing}"

cython_installation_sha256="$("$BUILD_PYTHON" - <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

distribution = importlib.metadata.distribution("Cython")
digest = hashlib.sha256()
count = 0
for relative in sorted(distribution.files or (), key=lambda item: str(item)):
    name = str(relative)
    if "__pycache__" in name or name.endswith((".pyc", ".pyo")):
        continue
    path = Path(distribution.locate_file(relative))
    if not path.is_file():
        continue
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
    digest.update(b"\0")
    count += 1
if count == 0:
    raise SystemExit("Cython distribution contains no hashable files")
print(digest.hexdigest())
PY
)"

target_metadata="$("$TARGET_PYTHON" -c '
import platform
import sys
import sysconfig
try:
    import numpy
    numpy_version = numpy.__version__
except ImportError:
    numpy_version = "unavailable"
values = (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    platform.machine(),
    str(sysconfig.get_config_var("SOABI") or ""),
    str(sysconfig.get_config_var("EXT_SUFFIX") or ""),
    str(sysconfig.get_path("include") or ""),
    str(sysconfig.get_path("platinclude") or ""),
    numpy_version,
)
print("\t".join(values))
')"
IFS=$'\t' read -r target_version target_arch target_soabi extension_suffix include_dir platinclude_dir target_numpy_version <<<"$target_metadata"

[[ "$target_version" == 3.12.* ]] || die "Target must be CPython 3.12; found $target_version"
[[ "$target_arch" == "arm64" ]] || die "Target Python must be arm64; found $target_arch"
[[ "$target_soabi" == "cpython-312-darwin" ]] || die "Unexpected target SOABI: $target_soabi"
[[ "$extension_suffix" == ".cpython-312-darwin.so" ]] || die "Unexpected extension suffix: $extension_suffix"
require_file "$include_dir/Python.h" "Target Python.h"
require_file "$platinclude_dir/Python.h" "Target platform Python.h"
target_python_sha256="$(shasum -a 256 "$TARGET_PYTHON" | awk '{print $1}')"
target_python_header_sha256="$(shasum -a 256 "$include_dir/Python.h" | awk '{print $1}')"

for module in "${NATIVE_MODULES[@]}"; do
  require_file "$PIPELINE_SRC_DIR/$module.py" "Native module source"
done

DIST_ROOT="$PROJECT_ROOT/dist"
mkdir -p "$DIST_ROOT"
DIST_ROOT="$(cd "$DIST_ROOT" && pwd)"
output_parent="$(dirname "$OUTPUT_DIR")"
output_name="$(basename "$OUTPUT_DIR")"
[[ -n "$output_name" && "$output_name" != "." && "$output_name" != ".." ]] || die "Unsafe output directory: $OUTPUT_DIR"
[[ -d "$output_parent" ]] || die "Output parent must already exist: $output_parent"
output_parent="$(cd "$output_parent" && pwd)"
OUTPUT_DIR="$output_parent/$output_name"
[[ "$output_parent" == "$DIST_ROOT" ]] || die "Output must be a direct child of $DIST_ROOT"
[[ ! -L "$OUTPUT_DIR" ]] || die "Output directory must not be a symbolic link: $OUTPUT_DIR"

source_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)"
source_tag="$(git -C "$PROJECT_ROOT" describe --tags --exact-match HEAD 2>/dev/null || true)"
source_dirty="0"
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=normal 2>/dev/null || true)" ]]; then
  source_dirty="1"
fi
repository_url="$(git -C "$PROJECT_ROOT" config --get remote.origin.url 2>/dev/null || true)"
if [[ "$repository_url" == /* ]]; then
  repository_url=""
fi
build_recipe_sha256="$(shasum -a 256 "$SCRIPT_DIR/build_native_pipeline.sh" | awk '{print $1}')"
verifier_sha256="$(shasum -a 256 "$SCRIPT_DIR/verify_native_pipeline.py" | awk '{print $1}')"
native_lock_sha256="$(shasum -a 256 "$SCRIPT_DIR/requirements-native-pipeline.lock" | awk '{print $1}')"
build_inputs_tracked="1"
for build_input in \
  build/build_native_pipeline.sh \
  build/verify_native_pipeline.py \
  build/requirements-native-pipeline.lock; do
  if ! git -C "$PROJECT_ROOT" ls-files --error-unmatch "$build_input" >/dev/null 2>&1; then
    build_inputs_tracked="0"
  fi
done

BUILD_ROOT="$(mktemp -d "$output_parent/.native_pipeline_build.XXXXXX")"
PAYLOAD_ROOT="$BUILD_ROOT/payload"
PAYLOAD_PIPELINE="$PAYLOAD_ROOT/pipeline"
C_ROOT="$BUILD_ROOT/generated-c"
SNAPSHOT_PIPELINE="$BUILD_ROOT/source/pipeline"
PREVIOUS_OUTPUT=""
FAILED_OUTPUT=""
PUBLISH_COMPLETE=0
mkdir -p "$PAYLOAD_PIPELINE" "$C_ROOT" "$(dirname "$SNAPSHOT_PIPELINE")"

cleanup() {
  local exit_status=$?
  set +e
  if [[ "$PUBLISH_COMPLETE" != "1" && -n "$PREVIOUS_OUTPUT" && -e "$PREVIOUS_OUTPUT" ]]; then
    if [[ -e "$OUTPUT_DIR" ]]; then
      FAILED_OUTPUT="$output_parent/.${output_name}.failed.$$"
      /bin/mv "$OUTPUT_DIR" "$FAILED_OUTPUT"
    fi
    /bin/mv "$PREVIOUS_OUTPUT" "$OUTPUT_DIR"
    if [[ -n "$FAILED_OUTPUT" && -e "$FAILED_OUTPUT" ]]; then
      rm -rf "$FAILED_OUTPUT"
    fi
  fi
  rm -rf "$BUILD_ROOT"
  return "$exit_status"
}
trap cleanup EXIT

log "Freezing pipeline source snapshot"
cp -R "$PIPELINE_SRC_DIR" "$SNAPSHOT_PIPELINE"
find "$SNAPSHOT_PIPELINE" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$SNAPSHOT_PIPELINE" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
for module in "${NATIVE_MODULES[@]}"; do
  require_file "$SNAPSHOT_PIPELINE/$module.py" "Snapshotted native module source"
done

log "Build Python: $BUILD_PYTHON (Cython $cython_version)"
log "Target Python: $TARGET_PYTHON ($target_version, $target_soabi, $target_arch)"
log "Pipeline source: $PIPELINE_SRC_DIR (frozen before compilation)"
log "Output: $OUTPUT_DIR"

for module in "${NATIVE_MODULES[@]}"; do
  c_path="$C_ROOT/$module.c"
  binary_path="$PAYLOAD_PIPELINE/$module$extension_suffix"

  log "Cythonizing $module"
  (
    cd "$BUILD_ROOT/source"
    "$BUILD_PYTHON" -m cython \
      -3 \
      -X binding=True \
      -X annotation_typing=False \
      -X infer_types=False \
      --output-file "$c_path" \
      "pipeline/$module.py"
  )

  log "Compiling $module -> $(basename "$binary_path")"
  /usr/bin/clang \
    -bundle \
    -undefined dynamic_lookup \
    -O2 \
    -DNDEBUG \
    -fPIC \
    -arch arm64 \
    "-mmacosx-version-min=$MACOS_MIN_VERSION" \
    -I "$include_dir" \
    -I "$platinclude_dir" \
    -o "$binary_path" \
    "$c_path"

  actual_archs="$(/usr/bin/lipo -archs "$binary_path")"
  [[ "$actual_archs" == "arm64" ]] || die "$module is not thin arm64: $actual_archs"
  /usr/bin/file "$binary_path" | /usr/bin/grep -q "Mach-O 64-bit bundle arm64" || die "$module is not a Mach-O arm64 bundle"
  /usr/bin/nm -gU "$binary_path" | /usr/bin/grep -q "_PyInit_$module" || die "$module has no matching PyInit symbol"
  linkage_output="$(/usr/bin/otool -L "$binary_path")" || die "otool -L failed for $module"
  while read -r dependency _remainder; do
    [[ -n "$dependency" ]] || continue
    case "$dependency" in
      /usr/lib/*|/System/Library/*)
        ;;
      *)
        die "$module contains a non-system dynamic dependency: $dependency"
        ;;
    esac
  done < <(printf '%s\n' "$linkage_output" | /usr/bin/sed -n '2,$p')
  load_commands="$(/usr/bin/otool -l "$binary_path")" || die "otool -l failed for $module"
  if printf '%s\n' "$load_commands" | /usr/bin/grep -q 'cmd LC_RPATH'; then
    die "$module must not contain LC_RPATH entries"
  fi

  /usr/bin/codesign --force --sign "$CODESIGN_IDENTITY" "$binary_path"
  /usr/bin/codesign --verify --strict "$binary_path"
done

log "Verifying native imports and representative source/native fixture equivalence"
"$BUILD_PYTHON" "$SCRIPT_DIR/verify_native_pipeline.py" \
  --python "$TARGET_PYTHON" \
  --source-dir "$SNAPSHOT_PIPELINE" \
  --native-dir "$PAYLOAD_PIPELINE" \
  --expected-modules "${NATIVE_MODULES[@]}"

clang_version="$(/usr/bin/clang --version | /usr/bin/sed -n '1p')"
signing_mode="ad_hoc"
if [[ "$CODESIGN_IDENTITY" != "-" ]]; then
  signing_mode="explicit_identity"
fi
manifest_path="$PAYLOAD_PIPELINE/native-pipeline-manifest.json"

"$BUILD_PYTHON" - \
  "$SNAPSHOT_PIPELINE" \
  "$PAYLOAD_PIPELINE" \
  "$manifest_path" \
  "$extension_suffix" \
  "$target_version" \
  "$target_soabi" \
  "$target_arch" \
  "$target_numpy_version" \
  "$target_python_sha256" \
  "$target_python_header_sha256" \
  "$MACOS_MIN_VERSION" \
  "$cython_version" \
  "$build_python_version" \
  "$build_python_arch" \
  "$cython_installation_sha256" \
  "$clang_version" \
  "$source_commit" \
  "$source_tag" \
  "$source_dirty" \
  "$repository_url" \
  "$SOURCE_ARCHIVE_SHA256" \
  "$build_recipe_sha256" \
  "$verifier_sha256" \
  "$native_lock_sha256" \
  "$build_inputs_tracked" \
  "$signing_mode" \
  "${NATIVE_MODULES[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


(
    source_root_arg,
    binary_root_arg,
    manifest_arg,
    extension_suffix,
    python_version,
    soabi,
    arch,
    numpy_version,
    target_python_sha256,
    target_python_header_sha256,
    macos_min,
    cython_version,
    build_python_version,
    build_python_arch,
    cython_installation_sha256,
    clang_version,
    commit_sha,
    source_tag,
    dirty_flag,
    repository_url,
    source_archive_sha256,
    build_recipe_sha256,
    verifier_sha256,
    native_lock_sha256,
    build_inputs_tracked_flag,
    signing_mode,
    *modules,
) = sys.argv[1:]

source_root = Path(source_root_arg)
binary_root = Path(binary_root_arg)
manifest_path = Path(manifest_arg)
dirty = dirty_flag == "1"
module_records = []
for module in modules:
    source_path = source_root / f"{module}.py"
    binary_path = binary_root / f"{module}{extension_suffix}"
    source_sha256 = sha256_file(source_path)
    module_records.append(
        {
            "import_name": module,
            "source_paths": [f"pipeline/{module}.py"],
            "source_set_sha256": source_sha256,
            "binary_path": f"pipeline/{binary_path.name}",
            "binary_sha256": sha256_file(binary_path),
            "size_bytes": binary_path.stat().st_size,
            "format": "Mach-O bundle",
            "archs": [arch],
        }
    )

blocking_reasons = ["technical_poc_not_dmg_release"]
if dirty:
    blocking_reasons.append("dirty_source_checkout")
if not source_archive_sha256:
    blocking_reasons.append("source_archive_sha256_missing")
if not repository_url:
    blocking_reasons.append("public_source_repository_url_missing")
if not source_tag:
    blocking_reasons.append("source_tag_missing")
if build_inputs_tracked_flag != "1":
    blocking_reasons.append("build_inputs_untracked")
if signing_mode == "ad_hoc":
    blocking_reasons.append("developer_id_signature_missing")
else:
    blocking_reasons.append("developer_id_signature_not_verified")
blocking_reasons.extend(
    (
        "notarization_missing",
        "legal_notice_bundle_missing",
        "reproducible_build_not_established",
        "app_embedding_requires_final_resign_and_rehash",
    )
)

manifest = {
    "schema": "starun.native-pipeline-build.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "distribution_intent": "technical_poc",
    "distribution_scope": "internal_only",
    "project": {
        "name": "Starun",
        "license_expression": "GPL-3.0-only",
    },
    "source": {
        "repository_url": repository_url or None,
        "commit_sha": commit_sha or None,
        "tag": source_tag or None,
        "clean": not dirty,
        "archive_sha256": source_archive_sha256.lower() or None,
        "build_recipe_path": "build/build_native_pipeline.sh",
        "build_recipe_sha256": build_recipe_sha256,
        "build_inputs_tracked": build_inputs_tracked_flag == "1",
        "build_inputs": [
            {
                "path": "build/build_native_pipeline.sh",
                "sha256": build_recipe_sha256,
            },
            {
                "path": "build/verify_native_pipeline.py",
                "sha256": verifier_sha256,
            },
            {
                "path": "build/requirements-native-pipeline.lock",
                "sha256": native_lock_sha256,
            },
        ],
        "snapshot": {
            "frozen_before_compilation": True,
            "module_hashes_from_snapshot": True,
        },
    },
    "target": {
        "python_version": python_version,
        "soabi": soabi,
        "extension_suffix": extension_suffix,
        "arch": arch,
        "numpy_version": numpy_version,
        "runtime_executable_sha256": target_python_sha256,
        "python_header_sha256": target_python_header_sha256,
        "macos_min_version": macos_min,
    },
    "compiler": {
        "cython_version": cython_version,
        "cython_installation_sha256": cython_installation_sha256,
        "build_python_version": build_python_version,
        "build_python_arch": build_python_arch,
        "clang": clang_version,
        "directives": {
            "language_level": 3,
            "annotation_typing": False,
            "infer_types": False,
            "binding": True,
        },
    },
    "native_scope": {
        "policy": "declared_project_modules",
        "modules": module_records,
        "forbidden_fallbacks": ["matching .py", "matching .pyc"],
    },
    "payload": {
        "inventory_policy": "exact_files_except_manifest",
        "manifest_path": "pipeline/native-pipeline-manifest.json",
        "files": [
            {
                "path": record["binary_path"],
                "sha256": record["binary_sha256"],
                "size_bytes": record["size_bytes"],
            }
            for record in module_records
        ],
    },
    "signing": {
        "mode": signing_mode,
        "verified": True,
        "verified_scope": "declared_native_modules",
        "developer_id_verified": False,
        "hardened_runtime": False,
        "notarized": False,
    },
    "verification": {
        "profile": "starun.native-pipeline-representative-fixture.v1",
        "thin_arm64": True,
        "cpython_312_import_smoke": True,
        "representative_source_native_fixture_equivalence": True,
        "covered_functions": {
            "stage3_contract": [
                "stage3_gate_thresholds",
                "stage3_static_contract_manifest",
            ],
            "background_sampling": ["analyze_directional_pattern_noise"],
            "stage4_auto_reference": ["evaluate_auto_local_reference"],
            "local_adjustments": [
                "build_local_masks",
                "apply_monotonic_curve",
            ],
            "stage9_quality": [
                "screen_blend",
                "unscreen_layer",
                "stage9_scale_radius",
            ],
        },
    },
    "reproducible_payload": False,
    "poc_accepted": True,
    "public_release_eligible": False,
    "blocking_reasons": blocking_reasons,
}
canonical = json.dumps(
    manifest,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
manifest["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

PREVIOUS_OUTPUT="$output_parent/.${output_name}.previous.$$"
[[ ! -e "$PREVIOUS_OUTPUT" ]] || die "Temporary output backup already exists: $PREVIOUS_OUTPUT"
if [[ -e "$OUTPUT_DIR" ]]; then
  /bin/mv "$OUTPUT_DIR" "$PREVIOUS_OUTPUT"
fi
if ! /bin/mv "$PAYLOAD_ROOT" "$OUTPUT_DIR"; then
  die "Failed to publish native pipeline output"
fi
PUBLISH_COMPLETE=1
if [[ -e "$PREVIOUS_OUTPUT" ]]; then
  rm -rf "$PREVIOUS_OUTPUT"
fi
PREVIOUS_OUTPUT=""

log "Native pipeline PoC accepted"
log "Manifest: $OUTPUT_DIR/pipeline/native-pipeline-manifest.json"
log "Public release eligible: false"
