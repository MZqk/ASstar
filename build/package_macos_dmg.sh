#!/usr/bin/env bash
set -euo pipefail

# Package one already-built Starun.app into a verified DMG.
#
# Local mode is intentionally ad-hoc and internal-only. Formal mode requires an
# already hardened/Developer-ID-signed App, notarizes and staples that App via a
# ZIP submission, then signs, notarizes, and staples the containing DMG.

MODE=""
MODE_EXPLICIT=0
LOCAL_ADHOC=0
APP_PATH=""
OUTPUT_DMG=""
OUTPUT_SHA=""
OUTPUT_RELEASE=""
VOLUME_NAME=""
CODESIGN_IDENTITY=""
NOTARY_PROFILE=""
NOTARY_TIMEOUT="1h"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
NATIVE_MANAGER="$SCRIPT_DIR/manage_native_pipeline_bundle.py"

NATIVE_MODULES=(
  stage3_contract
  background_sampling
  stage4_auto_reference
  local_adjustments
  stage9_quality
)
NATIVE_SUFFIX=".cpython-312-darwin.so"

BUILD_ROOT=""
MOUNT_DIR=""
MOUNT_DEVICE=""
MOUNT_IMAGE=""
MOUNTED=0
OUTPUT_PARENT=""
VERIFY_PYTHON=""
VERIFIED_APP_TEAM=""
VERIFIED_APP_CDHASH=""
VERIFIED_DMG_TEAM=""
VERIFIED_DMG_CDHASH=""
MOUNTED_RUNTIME_MANIFEST_SHA=""
SIGNING_TEAM=""
APP_NOTARY_ID=""
DMG_NOTARY_ID=""
FORMAL_GATES_COMPLETE=0
PUBLISH_ACTIVE=0
PUBLISH_NEW_DMG=0
PUBLISH_NEW_SHA=0
PUBLISH_NEW_RELEASE=0
PUBLISH_BACKUP_DMG=""
PUBLISH_BACKUP_SHA=""
PUBLISH_BACKUP_RELEASE=""
LOCK_DIR=""
LOCK_ACQUIRED=0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --local-adhoc               Explicit ad-hoc internal DMG mode
  --mode local|formal          Compatibility alias for explicit mode selection
  --app PATH                   Signed .app input (default: release/Starun.app)
  --output PATH                Final .dmg path (default: release/<AppName>.dmg)
  --volume-name NAME           Mounted volume name (default: Install <AppName>)
  --codesign-identity ID       Full Developer ID Application identity in formal mode
  --notary-profile PROFILE     notarytool Keychain profile in formal mode
  --notary-timeout DURATION    notarytool wait timeout (default: 1h)
  --help                       Show this help

Safety:
  The output, its .sha256 file, and its .release.json evidence must be under
  this checkout's dist/ or release/ directory. The input App is never modified;
  formal stapling happens only on a staging copy.
EOF
}

die() {
  echo "[DMG][ERROR] $*" >&2
  exit 1
}

log() {
  echo "[DMG] $*"
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" && ! -L "$path" ]] || die "$label is missing or is a symlink: $path"
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" && ! -L "$path" ]] || die "$label is missing or is a symlink: $path"
}

require_executable() {
  local path="$1"
  local label="$2"
  [[ -x "$path" ]] || die "$label is not executable: $path"
}

require_command_path() {
  local path="$1"
  local label="$2"
  [[ -x "$path" ]] || die "$label is unavailable: $path"
}

mount_state() {
  local match_mode="${1:-any}"
  local info_plist=""

  [[ -n "$BUILD_ROOT" && -d "$BUILD_ROOT" && -n "$MOUNT_DIR" \
     && -n "$MOUNT_IMAGE" ]] || return 2
  info_plist="$BUILD_ROOT/hdiutil-info.plist"
  /usr/bin/hdiutil info -plist >"$info_plist" 2>/dev/null || return 2
  "$VERIFY_PYTHON" - \
    "$info_plist" "$MOUNT_IMAGE" "$MOUNT_DIR" "$MOUNT_DEVICE" "$match_mode" <<'PY'
import os
import plistlib
import re
import sys
from pathlib import Path

info_path = Path(sys.argv[1])
image_path = os.path.realpath(sys.argv[2])
mount_point = sys.argv[3]
device = sys.argv[4]
match_mode = sys.argv[5]
try:
    with info_path.open("rb") as handle:
        payload = plistlib.load(handle)
except (OSError, plistlib.InvalidFileException):
    raise SystemExit(2)

try:
    matching_images = []
    images = payload.get("images", [])
    if not isinstance(images, list):
        raise TypeError("images is not a list")
    for image in images:
        if not isinstance(image, dict):
            raise TypeError("image entry is not a dictionary")
        candidate = str(image.get("image-path") or "")
        if candidate and os.path.realpath(candidate) == image_path:
            matching_images.append(image)
except (AttributeError, TypeError, ValueError):
    raise SystemExit(2)

if not matching_images:
    # Only total absence of this unique staging image proves it is detached.
    raise SystemExit(1)

if match_mode == "any":
    raise SystemExit(0)

entities = []
for image in matching_images:
    image_entities = image.get("system-entities", [])
    if not isinstance(image_entities, list):
        raise SystemExit(2)
    for entity in image_entities:
        if not isinstance(entity, dict):
            raise SystemExit(2)
        entities.append(entity)
        entity_mount = str(entity.get("mount-point") or "")
        entity_device = str(entity.get("dev-entry") or "")
        if match_mode == "pair":
            if entity_mount == mount_point and device and entity_device == device:
                raise SystemExit(0)

if match_mode == "device":
    devices = [str(entity.get("dev-entry") or "") for entity in entities]
    devices = [value for value in devices if re.fullmatch(r"/dev/disk[0-9]+(?:s[0-9]+)?", value)]
    base_devices = [value for value in devices if re.fullmatch(r"/dev/disk[0-9]+", value)]
    if base_devices or devices:
        print((base_devices or devices)[0])
        raise SystemExit(0)

# The image is attached, but the requested relation/device was not observable.
raise SystemExit(3)
PY
}

detach_mount() {
  local detach_target=""
  local discovered_device=""
  local state=2

  if [[ "$MOUNTED" != "1" || -z "$MOUNT_DIR" ]]; then
    return 0
  fi

  if mount_state any; then
    state=0
  else
    state=$?
  fi
  if [[ "$state" == "1" ]]; then
    MOUNTED=0
    MOUNT_DEVICE=""
    return 0
  fi
  if [[ "$state" != "0" ]]; then
    echo "[DMG][WARN] Unable to query the temporary image attachment state: $MOUNT_IMAGE" >&2
    return 1
  fi

  if discovered_device="$(mount_state device)"; then
    MOUNT_DEVICE="$discovered_device"
  else
    echo "[DMG][WARN] Unable to identify the temporary image device: $MOUNT_IMAGE" >&2
    return 1
  fi
  [[ "$MOUNT_DEVICE" == /dev/disk* ]] || return 1
  detach_target="$MOUNT_DEVICE"
  /usr/bin/hdiutil detach "$detach_target" >/dev/null 2>&1 || true
  if mount_state any; then
    state=0
  else
    state=$?
  fi
  if [[ "$state" == "1" ]]; then
    MOUNTED=0
    MOUNT_DEVICE=""
    return 0
  fi

  if discovered_device="$(mount_state device)"; then
    MOUNT_DEVICE="$discovered_device"
    detach_target="$MOUNT_DEVICE"
  else
    echo "[DMG][WARN] Unable to rediscover the temporary image device: $MOUNT_IMAGE" >&2
    return 1
  fi
  /usr/bin/hdiutil detach "$detach_target" -force >/dev/null 2>&1 || true
  if mount_state any; then
    state=0
  else
    state=$?
  fi
  if [[ "$state" == "1" ]]; then
    MOUNTED=0
    MOUNT_DEVICE=""
    return 0
  fi

  echo "[DMG][WARN] Failed to detach or prove detached temporary mount: $MOUNT_DIR" >&2
  return 1
}

cleanup() {
  local status=$?
  local rollback_failed=0
  set +e
  detach_mount
  if [[ "$MOUNTED" == "1" ]]; then
    echo "[DMG][WARN] Leaving staging directory because its DMG is still mounted: $BUILD_ROOT" >&2
    [[ "$status" -ne 0 ]] || status=1
    return "$status"
  fi
  if [[ "$PUBLISH_ACTIVE" == "1" ]]; then
    if [[ "$PUBLISH_NEW_DMG" == "1" ]] && ! /bin/rm -f -- "$OUTPUT_DMG"; then
      echo "[DMG][ERROR] Failed to remove partially published DMG: $OUTPUT_DMG" >&2
      rollback_failed=1
    fi
    if [[ "$PUBLISH_NEW_SHA" == "1" ]] && ! /bin/rm -f -- "$OUTPUT_SHA"; then
      echo "[DMG][ERROR] Failed to remove partially published checksum: $OUTPUT_SHA" >&2
      rollback_failed=1
    fi
    if [[ "$PUBLISH_NEW_RELEASE" == "1" ]] && ! /bin/rm -f -- "$OUTPUT_RELEASE"; then
      echo "[DMG][ERROR] Failed to remove partially published release evidence: $OUTPUT_RELEASE" >&2
      rollback_failed=1
    fi
    if [[ -n "$PUBLISH_BACKUP_DMG" \
          && ( -e "$PUBLISH_BACKUP_DMG" || -L "$PUBLISH_BACKUP_DMG" ) ]]; then
      if [[ ! -f "$PUBLISH_BACKUP_DMG" || -L "$PUBLISH_BACKUP_DMG" ]] \
          || ! /bin/mv "$PUBLISH_BACKUP_DMG" "$OUTPUT_DMG"; then
        echo "[DMG][ERROR] Failed to restore previous DMG; backup preserved at: $PUBLISH_BACKUP_DMG" >&2
        rollback_failed=1
      fi
    fi
    if [[ -n "$PUBLISH_BACKUP_SHA" \
          && ( -e "$PUBLISH_BACKUP_SHA" || -L "$PUBLISH_BACKUP_SHA" ) ]]; then
      if [[ ! -f "$PUBLISH_BACKUP_SHA" || -L "$PUBLISH_BACKUP_SHA" ]] \
          || ! /bin/mv "$PUBLISH_BACKUP_SHA" "$OUTPUT_SHA"; then
        echo "[DMG][ERROR] Failed to restore previous checksum; backup preserved at: $PUBLISH_BACKUP_SHA" >&2
        rollback_failed=1
      fi
    fi
    if [[ -n "$PUBLISH_BACKUP_RELEASE" \
          && ( -e "$PUBLISH_BACKUP_RELEASE" || -L "$PUBLISH_BACKUP_RELEASE" ) ]]; then
      if [[ ! -f "$PUBLISH_BACKUP_RELEASE" || -L "$PUBLISH_BACKUP_RELEASE" ]] \
          || ! /bin/mv "$PUBLISH_BACKUP_RELEASE" "$OUTPUT_RELEASE"; then
        echo "[DMG][ERROR] Failed to restore previous release evidence; backup preserved at: $PUBLISH_BACKUP_RELEASE" >&2
        rollback_failed=1
      fi
    fi
    if [[ "$rollback_failed" == "1" ]]; then
      echo "[DMG][ERROR] Publication rollback was incomplete; preserving staging and lock for recovery." >&2
      [[ "$status" -ne 0 ]] || status=1
      return "$status"
    fi
  fi
  if [[ -n "$BUILD_ROOT" && -n "$OUTPUT_PARENT" ]]; then
    case "$BUILD_ROOT" in
      "$OUTPUT_PARENT"/.starun_dmg_package.*)
        /bin/rm -rf -- "$BUILD_ROOT"
        ;;
      *)
        echo "[DMG][WARN] Refusing to clean unexpected staging path: $BUILD_ROOT" >&2
        ;;
    esac
  fi
  if [[ "$LOCK_ACQUIRED" == "1" && -n "$LOCK_DIR" ]]; then
    /bin/rmdir "$LOCK_DIR" >/dev/null 2>&1 \
      || echo "[DMG][WARN] Failed to remove packaging lock: $LOCK_DIR" >&2
  fi
  return "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      MODE="$2"
      MODE_EXPLICIT=1
      shift 2
      ;;
    --local-adhoc)
      LOCAL_ADHOC=1
      shift
      ;;
    --app)
      [[ $# -ge 2 ]] || die "--app requires a value"
      APP_PATH="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || die "--output requires a value"
      OUTPUT_DMG="$2"
      shift 2
      ;;
    --volume-name)
      [[ $# -ge 2 ]] || die "--volume-name requires a value"
      VOLUME_NAME="$2"
      shift 2
      ;;
    --codesign-identity)
      [[ $# -ge 2 ]] || die "--codesign-identity requires a value"
      CODESIGN_IDENTITY="$2"
      shift 2
      ;;
    --notary-profile)
      [[ $# -ge 2 ]] || die "--notary-profile requires a value"
      NOTARY_PROFILE="$2"
      shift 2
      ;;
    --notary-timeout)
      [[ $# -ge 2 ]] || die "--notary-timeout requires a value"
      NOTARY_TIMEOUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if [[ "$LOCAL_ADHOC" == "1" ]]; then
  if [[ "$MODE_EXPLICIT" == "1" && "$MODE" != "local" ]]; then
    die "--local-adhoc conflicts with --mode $MODE"
  fi
  [[ -z "$CODESIGN_IDENTITY" && -z "$NOTARY_PROFILE" ]] \
    || die "--local-adhoc is mutually exclusive with formal signing parameters"
  MODE="local"
elif [[ "$MODE_EXPLICIT" == "1" ]]; then
  case "$MODE" in
    local|formal) ;;
    *) die "--mode must be local or formal: $MODE" ;;
  esac
elif [[ -n "$CODESIGN_IDENTITY" || -n "$NOTARY_PROFILE" ]]; then
  MODE="formal"
else
  die "select --local-adhoc or provide both --codesign-identity and --notary-profile"
fi

if [[ "$MODE" == "local" ]]; then
  [[ -z "$NOTARY_PROFILE" ]] || die "local mode does not accept --notary-profile"
  [[ -z "$CODESIGN_IDENTITY" || "$CODESIGN_IDENTITY" == "-" ]] \
    || die "local mode only supports the ad-hoc '-' identity"
  CODESIGN_IDENTITY="-"
else
  [[ -n "$CODESIGN_IDENTITY" && "$CODESIGN_IDENTITY" != "-" ]] \
    || die "formal mode requires --codesign-identity"
  [[ "$CODESIGN_IDENTITY" == "Developer ID Application: "* ]] \
    || die "formal mode requires the full Developer ID Application identity"
  if [[ "$CODESIGN_IDENTITY" =~ \(([A-Z0-9]{10})\)$ ]]; then
    SIGNING_TEAM="${BASH_REMATCH[1]}"
  else
    die "Developer ID identity must end with its 10-character Team ID"
  fi
  [[ -n "$NOTARY_PROFILE" ]] || die "formal mode requires --notary-profile"
fi

[[ "$NOTARY_TIMEOUT" =~ ^[1-9][0-9]*(s|m|h)?$ ]] \
  || die "invalid --notary-timeout: $NOTARY_TIMEOUT"

APP_PATH="${APP_PATH:-$PROJECT_ROOT/release/Starun.app}"
APP_PATH="${APP_PATH/#\~/$HOME}"
app_parent_raw="$(dirname "$APP_PATH")"
[[ -d "$app_parent_raw" ]] || die "App parent directory is missing: $app_parent_raw"
app_parent="$(cd "$app_parent_raw" && pwd -P)"
APP_PATH="$app_parent/$(basename "$APP_PATH")"
require_dir "$APP_PATH" "Input App"
[[ "$APP_PATH" == *.app ]] || die "input must be a .app bundle: $APP_PATH"

APP_BASENAME="$(basename "$APP_PATH")"
APP_NAME="${APP_BASENAME%.app}"
[[ -n "$APP_NAME" ]] || die "unable to derive App name from: $APP_BASENAME"

OUTPUT_DMG="${OUTPUT_DMG:-$PROJECT_ROOT/release/$APP_NAME.dmg}"
OUTPUT_DMG="${OUTPUT_DMG/#\~/$HOME}"
OUTPUT_NAME="$(basename "$OUTPUT_DMG")"
[[ -n "$OUTPUT_NAME" && "$OUTPUT_NAME" != "." && "$OUTPUT_NAME" != ".." ]] \
  || die "unsafe output filename: $OUTPUT_NAME"
[[ "$OUTPUT_NAME" == *.dmg ]] || die "output must end in .dmg: $OUTPUT_NAME"

[[ ! -L "$PROJECT_ROOT/dist" && ! -L "$PROJECT_ROOT/release" ]] \
  || die "dist/ and release/ must not be symbolic links"
/bin/mkdir -p "$PROJECT_ROOT/dist" "$PROJECT_ROOT/release"
[[ ! -L "$PROJECT_ROOT/dist" && ! -L "$PROJECT_ROOT/release" ]] \
  || die "dist/ and release/ must not be symbolic links"
DIST_ROOT="$(cd "$PROJECT_ROOT/dist" && pwd -P)"
RELEASE_ROOT="$(cd "$PROJECT_ROOT/release" && pwd -P)"
[[ "$DIST_ROOT" == "$PROJECT_ROOT/dist" ]] \
  || die "dist/ resolves outside the canonical project root: $DIST_ROOT"
[[ "$RELEASE_ROOT" == "$PROJECT_ROOT/release" ]] \
  || die "release/ resolves outside the canonical project root: $RELEASE_ROOT"
output_parent_raw="$(dirname "$OUTPUT_DMG")"
[[ -d "$output_parent_raw" ]] \
  || die "output parent must already exist under dist/ or release/: $output_parent_raw"
OUTPUT_PARENT="$(cd "$output_parent_raw" && pwd -P)"
case "$OUTPUT_PARENT" in
  "$DIST_ROOT"|"$DIST_ROOT"/*|"$RELEASE_ROOT"|"$RELEASE_ROOT"/*) ;;
  *) die "output must be under $DIST_ROOT or $RELEASE_ROOT" ;;
esac
case "$OUTPUT_PARENT/" in
  "$APP_PATH/"*) die "output cannot be written inside the input App" ;;
esac
OUTPUT_DMG="$OUTPUT_PARENT/$OUTPUT_NAME"
OUTPUT_SHA="$OUTPUT_DMG.sha256"
OUTPUT_RELEASE="$OUTPUT_DMG.release.json"

for existing_output in "$OUTPUT_DMG" "$OUTPUT_SHA" "$OUTPUT_RELEASE"; do
  if [[ -e "$existing_output" || -L "$existing_output" ]]; then
    [[ -f "$existing_output" && ! -L "$existing_output" ]] \
      || die "refusing to replace a non-regular output or symlink: $existing_output"
  fi
done

VOLUME_NAME="${VOLUME_NAME:-Install $APP_NAME}"
[[ -n "$VOLUME_NAME" ]] || die "volume name must not be empty"
[[ "$VOLUME_NAME" != *$'\n'* && "$VOLUME_NAME" != */* && "$VOLUME_NAME" != *:* ]] \
  || die "volume name contains an unsafe character"

require_command_path /usr/bin/codesign "codesign"
require_command_path /usr/bin/ditto "ditto"
require_command_path /usr/bin/file "file"
require_command_path /usr/bin/hdiutil "hdiutil"
require_command_path /usr/bin/lipo "lipo"
require_command_path /usr/bin/plutil "plutil"
require_command_path /usr/bin/shasum "shasum"
require_command_path /usr/sbin/spctl "spctl"
if [[ "$MODE" == "formal" ]]; then
  require_command_path /usr/bin/xcrun "xcrun"
  require_command_path /usr/bin/security "security"
  identity_listing="$(/usr/bin/security find-identity -p codesigning -v 2>&1)"
  printf '%s\n' "$identity_listing" \
    | /usr/bin/grep -Fq "\"$CODESIGN_IDENTITY\"" \
    || die "requested Developer ID identity is not available in the signing keychain"
fi
require_file "$NATIVE_MANAGER" "Native runtime verifier"

if [[ -x "$PROJECT_ROOT/build/.venv/bin/python" ]]; then
  VERIFY_PYTHON="$PROJECT_ROOT/build/.venv/bin/python"
else
  VERIFY_PYTHON="$(command -v python3 || true)"
fi
require_executable "$VERIFY_PYTHON" "Verification Python"

codesign_detail() {
  local target="$1"
  /usr/bin/codesign -dvvv "$target" 2>&1
}

extract_team_identifier() {
  /usr/bin/sed -n 's/^TeamIdentifier=//p' | /usr/bin/sed -n '1p'
}

extract_cdhash() {
  /usr/bin/sed -n 's/^CDHash=//p' | /usr/bin/sed -n '1p'
}

verify_signature_mode() {
  local target="$1"
  local label="$2"
  local expected_team="$3"
  local detail=""
  local requirement=""
  local team=""

  if ! detail="$(codesign_detail "$target")"; then
    echo "$detail" >&2
    die "$label signing metadata is unavailable: $target"
  fi
  team="$(printf '%s\n' "$detail" | extract_team_identifier)"

  if [[ "$MODE" == "local" ]]; then
    printf '%s\n' "$detail" | /usr/bin/grep -Fxq "Signature=adhoc" \
      || die "$label must be ad-hoc signed in local mode: $target"
    [[ -z "$team" || "$team" == "not set" ]] \
      || die "$label unexpectedly has a TeamIdentifier in local mode: $target"
    return 0
  fi

  if ! requirement="$(/usr/bin/codesign -d -r- "$target" 2>&1)"; then
    echo "$requirement" >&2
    die "$label designated requirement is unavailable: $target"
  fi
  printf '%s\n' "$requirement" \
    | /usr/bin/grep -Fq 'certificate leaf[field.1.2.840.113635.100.6.1.13]' \
    || die "$label is not signed with a Developer ID Application certificate: $target"
  printf '%s\n' "$detail" | /usr/bin/grep -Eq '^CodeDirectory .*flags=.*\(runtime\)' \
    || die "$label lacks hardened runtime: $target"
  [[ -n "$team" && "$team" != "not set" ]] \
    || die "$label lacks a TeamIdentifier: $target"
  if [[ -n "$expected_team" && "$team" != "$expected_team" ]]; then
    die "$label TeamIdentifier mismatch: want=$expected_team got=$team target=$target"
  fi
  [[ "$team" == "$SIGNING_TEAM" ]] \
    || die "$label does not match the requested signing Team ID: want=$SIGNING_TEAM got=$team"
  printf '%s\n' "$team"
}

verify_siril_python_entitlements() {
  local app="$1"
  local python_root="$app/Contents/Resources/Siril.app/Contents/Frameworks/Python.framework/Versions/3.12"
  local py_bin="$python_root/bin/python3.12"
  local py_app_bin="$python_root/Resources/Python.app/Contents/MacOS/Python"

  require_executable "$py_bin" "Embedded Siril Python launcher"
  require_executable "$py_app_bin" "Embedded Siril Python.app launcher"
  "$VERIFY_PYTHON" - "$py_bin" "$py_app_bin" <<'PY'
import plistlib
import subprocess
import sys

required = (
    "com.apple.security.cs.allow-unsigned-executable-memory",
    "com.apple.security.cs.disable-library-validation",
)
for launcher in sys.argv[1:]:
    completed = subprocess.run(
        ["/usr/bin/codesign", "-d", "--entitlements", ":-", launcher],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).decode("utf-8", "replace")[-1000:]
        raise SystemExit(f"unable to read Siril Python entitlements: {launcher}: {detail}")
    output = completed.stdout + completed.stderr
    starts = [offset for offset in (output.find(b"<?xml"), output.find(b"<plist")) if offset >= 0]
    if not starts:
        raise SystemExit(f"Siril Python entitlement plist is missing: {launcher}")
    start = min(starts)
    end = output.find(b"</plist>", start)
    if end < 0:
        raise SystemExit(f"Siril Python entitlement plist is truncated: {launcher}")
    try:
        entitlements = plistlib.loads(output[start : end + len(b"</plist>")])
    except (plistlib.InvalidFileException, ValueError) as error:
        raise SystemExit(f"invalid Siril Python entitlement plist: {launcher}: {error}")
    if not isinstance(entitlements, dict):
        raise SystemExit(f"Siril Python entitlements are not a dictionary: {launcher}")
    invalid = [key for key in required if entitlements.get(key) is not True]
    if invalid:
        raise SystemExit(
            f"Siril Python launcher requires boolean true entitlements {invalid}: {launcher}"
        )
print("siril_python_entitlements_verified=" + ",".join(sys.argv[1:]))
PY
}

verify_all_nested_code() {
  local app="$1"
  local expected_team="$2"
  local siril_app="$app/Contents/Resources/Siril.app"
  local python_app="$siril_app/Contents/Frameworks/Python.framework/Versions/3.12/Resources/Python.app"
  local python_app_binary="$python_app/Contents/MacOS/Python"
  local code_path=""
  local file_kind=""
  local detail=""
  local nested_team=""
  local siril_team=""
  local siril_deep_verified=0

  [[ "$MODE" == "formal" ]] || return 0

  require_dir "$siril_app" "Embedded Siril App"
  if ! detail="$(/usr/bin/codesign --verify --deep --strict --verbose=2 "$siril_app" 2>&1)"; then
    echo "$detail" >&2
    die "embedded Siril App signature verification failed: $siril_app"
  fi
  siril_deep_verified=1
  siril_team="$(verify_signature_mode \
    "$siril_app" "Embedded Siril App" "$expected_team")"
  [[ "$siril_team" == "$expected_team" ]] \
    || die "embedded Siril App TeamIdentifier mismatch"

  while IFS= read -r -d '' code_path; do
    file_kind="$(/usr/bin/file -b "$code_path")"
    if ! printf '%s\n' "$file_kind" \
      | /usr/bin/grep -Eq 'Mach-O.*(executable|dynamically linked shared library|bundle)'; then
      continue
    fi
    if [[ "$code_path" == "$python_app_binary" ]]; then
      [[ "$siril_deep_verified" == "1" ]] \
        || die "Python.app exception requires a valid enclosing Siril seal"
      if ! detail="$(/usr/bin/codesign --verify --strict --ignore-resources \
          --verbose=2 "$code_path" 2>&1)"; then
        echo "$detail" >&2
        die "nested Python.app launcher signature verification failed: $code_path"
      fi
    elif ! detail="$(/usr/bin/codesign --verify --strict --verbose=2 "$code_path" 2>&1)"; then
      echo "$detail" >&2
      die "nested Mach-O signature verification failed: $code_path"
    fi
    nested_team="$(verify_signature_mode "$code_path" "Nested Mach-O" "$expected_team")"
    [[ "$nested_team" == "$expected_team" ]] \
      || die "nested Mach-O TeamIdentifier mismatch: $code_path"
  done < <(/usr/bin/find "$app/Contents" -type f -print0)

  while IFS= read -r -d '' code_path; do
    if [[ "$code_path" == "$python_app" && "$siril_deep_verified" == "1" ]]; then
      detail="$(/usr/bin/codesign --verify --strict --ignore-resources \
        --verbose=2 "$code_path" 2>&1)" || {
        echo "$detail" >&2
        die "nested Python.app signature verification failed: $code_path"
      }
    elif ! detail="$(/usr/bin/codesign --verify --strict --verbose=2 "$code_path" 2>&1)"; then
      echo "$detail" >&2
      die "nested code bundle signature verification failed: $code_path"
    fi
    nested_team="$(verify_signature_mode "$code_path" "Nested code bundle" "$expected_team")"
    [[ "$nested_team" == "$expected_team" ]] \
      || die "nested code bundle TeamIdentifier mismatch: $code_path"
  done < <(
    /usr/bin/find "$app/Contents" -depth -type d \
      \( -name '*.framework' -o -name '*.bundle' -o -name '*.xpc' \
         -o -name '*.appex' -o -name '*.app' \) -print0
  )
}

verify_bundle_symlinks() {
  local bundle_root="$1"
  "$VERIFY_PYTHON" - "$bundle_root" <<'PY' || die "App bundle contains invalid symbolic links"
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
if root.is_symlink() or not root.is_dir():
    raise SystemExit(f"App bundle root is missing or is a symlink: {root}")
resolved_root = root.resolve(strict=True)
invalid: list[str] = []
for directory, dirnames, filenames in os.walk(root, followlinks=False):
    for name in (*dirnames, *filenames):
        link = Path(directory) / name
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        if Path(target).is_absolute():
            invalid.append(f"absolute symlink: {link} -> {target}")
            continue
        try:
            resolved = link.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (FileNotFoundError, RuntimeError, ValueError):
            invalid.append(f"broken or escaping symlink: {link} -> {target}")
if invalid:
    for message in invalid:
        print(f"[DMG][ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)
PY
}

verify_app_bundle() {
  local app="$1"
  local info_plist="$app/Contents/Info.plist"
  local pipeline_dir="$app/Contents/Resources/pipeline"
  local target_python="$app/Contents/Resources/SirilPythonSeed/venv/bin/python3.12"
  local executable_name=""
  local main_binary=""
  local archs=""
  local detail=""
  local app_team=""
  local app_cdhash=""
  local module=""
  local binary=""
  local module_archs=""
  local module_team=""

  require_dir "$app" "App bundle"
  require_file "$info_plist" "App Info.plist"
  require_dir "$pipeline_dir" "Embedded pipeline"
  verify_bundle_symlinks "$app"
  require_executable "$target_python" "Embedded Siril CPython 3.12"

  executable_name="$(/usr/bin/plutil -extract CFBundleExecutable raw -o - "$info_plist")"
  [[ -n "$executable_name" && "$executable_name" != */* ]] \
    || die "invalid CFBundleExecutable in $info_plist"
  main_binary="$app/Contents/MacOS/$executable_name"
  require_executable "$main_binary" "App main executable"
  archs="$(/usr/bin/lipo -archs "$main_binary")"
  [[ "$archs" == "arm64" ]] || die "App main executable must be thin arm64: $archs"

  if ! detail="$(/usr/bin/codesign --verify --deep --strict --verbose=2 "$app" 2>&1)"; then
    echo "$detail" >&2
    die "App signature verification failed: $app"
  fi
  app_team="$(verify_signature_mode "$app" "App" "")"
  detail="$(codesign_detail "$app")"
  app_cdhash="$(printf '%s\n' "$detail" | extract_cdhash)"
  [[ "$app_cdhash" =~ ^[0-9A-Fa-f]{40,64}$ ]] \
    || die "App CDHash is missing or invalid: $app"

  verify_siril_python_entitlements "$app"

  "$VERIFY_PYTHON" "$NATIVE_MANAGER" verify \
    --pipeline-dir "$pipeline_dir" \
    --target-python "$target_python"

  for module in "${NATIVE_MODULES[@]}"; do
    binary="$pipeline_dir/$module$NATIVE_SUFFIX"
    require_file "$binary" "Embedded native module"
    module_archs="$(/usr/bin/lipo -archs "$binary")"
    [[ "$module_archs" == "arm64" ]] \
      || die "native module must be thin arm64: $module ($module_archs)"
    if ! detail="$(/usr/bin/codesign --verify --strict --verbose=2 "$binary" 2>&1)"; then
      echo "$detail" >&2
      die "native module signature verification failed: $binary"
    fi
    module_team="$(verify_signature_mode "$binary" "Native module $module" "$app_team")"
    if [[ "$MODE" == "formal" && "$module_team" != "$app_team" ]]; then
      die "native module TeamIdentifier mismatch: $module"
    fi
  done

  verify_all_nested_code "$app" "$app_team"

  VERIFIED_APP_TEAM="$app_team"
  VERIFIED_APP_CDHASH="$app_cdhash"
  log "Verified App/native runtime/imports: $app"
}

submit_for_notarization() {
  local artifact="$1"
  local label="$2"
  local response_path="$3"
  local artifact_kind="$4"
  local submission_id=""

  log "Submitting $label for notarization..."
  if ! /usr/bin/xcrun notarytool submit "$artifact" \
      --keychain-profile "$NOTARY_PROFILE" \
      --wait \
      --timeout "$NOTARY_TIMEOUT" \
      --output-format json >"$response_path"; then
    die "$label notarization submission failed"
  fi
  if ! submission_id="$("$VERIFY_PYTHON" - "$response_path" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"{label} notary response is invalid: {error}")
status = str(payload.get("status") or "")
submission_id = str(payload.get("id") or "")
if status != "Accepted" or not submission_id:
    raise SystemExit(
        f"{label} notarization was not accepted: status={status!r} id={submission_id!r}"
    )
print(submission_id)
PY
  )"; then
    die "$label notarization response validation failed"
  fi
  case "$artifact_kind" in
    app) APP_NOTARY_ID="$submission_id" ;;
    dmg) DMG_NOTARY_ID="$submission_id" ;;
    *) die "unknown notarization artifact kind: $artifact_kind" ;;
  esac
  log "Notarization accepted: $label ($submission_id)"
}

assess_app() {
  local app="$1"
  local output=""
  if ! output="$(/usr/sbin/spctl --assess --type execute --verbose=4 "$app" 2>&1)"; then
    printf '%s\n' "$output" >&2
    die "Gatekeeper rejected notarized App: $app"
  fi
}

verify_dmg_signature() {
  local dmg="$1"
  local detail=""
  local requirement=""
  local team=""
  local cdhash=""

  if ! detail="$(/usr/bin/codesign --verify --verbose=4 "$dmg" 2>&1)"; then
    echo "$detail" >&2
    die "DMG signature verification failed: $dmg"
  fi
  if ! detail="$(codesign_detail "$dmg")"; then
    echo "$detail" >&2
    die "DMG signing metadata is unavailable: $dmg"
  fi
  team="$(printf '%s\n' "$detail" | extract_team_identifier)"
  cdhash="$(printf '%s\n' "$detail" | extract_cdhash)"
  [[ "$cdhash" =~ ^[0-9A-Fa-f]{40,64}$ ]] \
    || die "DMG CDHash is missing or invalid: $dmg"
  if [[ "$MODE" == "local" ]]; then
    printf '%s\n' "$detail" | /usr/bin/grep -Fxq "Signature=adhoc" \
      || die "local DMG must be ad-hoc signed"
    [[ -z "$team" || "$team" == "not set" ]] \
      || die "local DMG unexpectedly has a TeamIdentifier"
    team=""
  else
    if ! requirement="$(/usr/bin/codesign -d -r- "$dmg" 2>&1)"; then
      echo "$requirement" >&2
      die "DMG designated requirement is unavailable"
    fi
    printf '%s\n' "$requirement" \
      | /usr/bin/grep -Fq 'certificate leaf[field.1.2.840.113635.100.6.1.13]' \
      || die "DMG is not signed with a Developer ID Application certificate"
    [[ "$team" == "$VERIFIED_APP_TEAM" ]] \
      || die "DMG/App TeamIdentifier mismatch: app=$VERIFIED_APP_TEAM dmg=$team"
    [[ "$team" == "$SIGNING_TEAM" ]] \
      || die "DMG does not match requested signing Team ID: want=$SIGNING_TEAM got=$team"
  fi
  VERIFIED_DMG_TEAM="$team"
  VERIFIED_DMG_CDHASH="$cdhash"
}

assess_dmg() {
  local dmg="$1"
  local output=""
  if ! output="$(/usr/sbin/spctl --assess --type open \
      --context context:primary-signature --verbose=4 "$dmg" 2>&1)"; then
    printf '%s\n' "$output" >&2
    die "Gatekeeper rejected notarized DMG: $dmg"
  fi
}

generate_release_sidecar() {
  local output_path="$1"
  local dmg_path="$2"
  local app_path="$3"
  local final_sha="$4"
  local mounted_runtime_sha="$5"

  "$VERIFY_PYTHON" - \
    "$output_path" "$dmg_path" "$app_path" "$final_sha" "$mounted_runtime_sha" \
    "$MODE" "$FORMAL_GATES_COMPLETE" "$VERIFIED_APP_TEAM" "$VERIFIED_APP_CDHASH" \
    "$VERIFIED_DMG_TEAM" "$VERIFIED_DMG_CDHASH" "$SIGNING_TEAM" \
    "$APP_NOTARY_ID" "$DMG_NOTARY_ID" "$OUTPUT_NAME" <<'PY'
import hashlib
import json
import os
import plistlib
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

(
    output_arg,
    dmg_arg,
    app_arg,
    final_sha,
    mounted_runtime_sha,
    mode,
    formal_gates_raw,
    app_team,
    app_cdhash,
    dmg_team,
    dmg_cdhash,
    signing_team,
    app_notary_id,
    dmg_notary_id,
    output_name,
) = sys.argv[1:]

output_path = Path(output_arg)
dmg_path = Path(dmg_arg)
app_path = Path(app_arg)
runtime_relative = Path("Contents/Resources/pipeline/native-pipeline-runtime-manifest.json")
runtime_path = app_path / runtime_relative
info_path = app_path / "Contents/Info.plist"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validated_strings(values):
    result = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise SystemExit(f"invalid runtime blocking reason: {value!r}")
        result.append(value)
    return result

if mode not in {"local", "formal"}:
    raise SystemExit(f"unsupported release mode: {mode}")
formal_gates_complete = formal_gates_raw == "1"
if not re.fullmatch(r"[0-9a-f]{64}", final_sha):
    raise SystemExit("invalid final DMG SHA-256")
if not dmg_path.is_file() or dmg_path.is_symlink() or sha256_file(dmg_path) != final_sha:
    raise SystemExit("final DMG SHA-256 changed before release manifest generation")
if not runtime_path.is_file() or runtime_path.is_symlink():
    raise SystemExit(f"runtime manifest is missing from staged App: {runtime_path}")
runtime_sha = sha256_file(runtime_path)
if runtime_sha != mounted_runtime_sha:
    raise SystemExit("mounted/staged runtime manifest SHA-256 mismatch")
try:
    runtime_manifest = json.loads(runtime_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"runtime manifest is unreadable: {error}")
if not isinstance(runtime_manifest, dict):
    raise SystemExit("runtime manifest root must be an object")
if runtime_manifest.get("schema") != "starun.native-pipeline-runtime.v1":
    raise SystemExit("unexpected runtime manifest schema")
runtime_payload_sha = str(runtime_manifest.get("manifest_payload_sha256") or "")
if not re.fullmatch(r"[0-9a-f]{64}", runtime_payload_sha):
    raise SystemExit("runtime manifest payload hash is invalid")
runtime_payload = dict(runtime_manifest)
runtime_payload.pop("manifest_payload_sha256", None)
canonical_runtime = json.dumps(
    runtime_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
if hashlib.sha256(canonical_runtime).hexdigest() != runtime_payload_sha:
    raise SystemExit("runtime manifest payload hash verification failed")
runtime_blockers_value = runtime_manifest.get("blocking_reasons")
if not isinstance(runtime_blockers_value, list):
    raise SystemExit("runtime manifest blocking_reasons must be an array")
runtime_blockers = validated_strings(runtime_blockers_value)

try:
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)
except (OSError, plistlib.InvalidFileException) as error:
    raise SystemExit(f"App Info.plist is unreadable: {error}")
if not isinstance(info, dict):
    raise SystemExit("App Info.plist root must be a dictionary")

for label, value in (("App", app_cdhash), ("DMG", dmg_cdhash)):
    if not re.fullmatch(r"[0-9A-Fa-f]{40,64}", value):
        raise SystemExit(f"{label} CDHash is invalid")

package_resolvable = {"app_notarization_missing", "dmg_notarization_missing"}
resolved_blockers = []
if mode == "formal":
    if not formal_gates_complete:
        raise SystemExit("formal release manifest requires all formal packaging gates")
    if not re.fullmatch(r"[A-Z0-9]{10}", signing_team):
        raise SystemExit("formal signing Team ID is invalid")
    if app_team != signing_team or dmg_team != signing_team:
        raise SystemExit("formal App/DMG Team ID mismatch during release manifest generation")
    for label, submission_id in (("App", app_notary_id), ("DMG", dmg_notary_id)):
        try:
            uuid.UUID(submission_id)
        except (ValueError, AttributeError) as error:
            raise SystemExit(f"{label} notarization submission ID is invalid") from error
    resolved_blockers = [
        reason for reason in runtime_blockers if reason in package_resolvable
    ]
    final_blockers = [
        reason for reason in runtime_blockers if reason not in package_resolvable
    ]
else:
    if formal_gates_complete or app_notary_id or dmg_notary_id:
        raise SystemExit("local release evidence contains unexpected formal gate state")
    if app_team or dmg_team or signing_team:
        raise SystemExit("local release evidence contains an unexpected Team ID")
    final_blockers = list(runtime_blockers)
    for reason in (
        "local_ad_hoc_distribution",
        "developer_id_signature_missing",
        "app_notarization_missing",
        "dmg_notarization_missing",
    ):
        if reason not in final_blockers:
            final_blockers.append(reason)

release_eligible = mode == "formal" and formal_gates_complete and not final_blockers
notary_status = "Accepted" if mode == "formal" else "not_submitted"
manifest = {
    "schema": "starun.macos-release.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "mode": mode,
    "distribution_scope": (
        "developer_id_notarized" if mode == "formal" else "internal_only"
    ),
    "release_eligible": release_eligible,
    "source_runtime_blocking_reasons": runtime_blockers,
    "resolved_blocking_reasons": resolved_blockers,
    "blocking_reasons": final_blockers,
    "artifacts": {
        "app": {
            "bundle_name": app_path.name,
            "bundle_id": str(info.get("CFBundleIdentifier") or ""),
            "short_version": str(info.get("CFBundleShortVersionString") or ""),
            "build_version": str(info.get("CFBundleVersion") or ""),
            "team_identifier": app_team or None,
            "cdhash": app_cdhash.lower(),
        },
        "dmg": {
            "filename": output_name,
            "size_bytes": dmg_path.stat().st_size,
            "sha256": final_sha,
            "team_identifier": dmg_team or None,
            "cdhash": dmg_cdhash.lower(),
        },
        "checksum": {"filename": output_name + ".sha256"},
        "native_runtime_manifest": {
            "path_in_app": str(Path(app_path.name) / runtime_relative),
            "schema": str(runtime_manifest.get("schema") or ""),
            "sha256": runtime_sha,
            "manifest_payload_sha256": runtime_payload_sha,
        },
    },
    "notarization": {
        "app": {
            "status": notary_status,
            "submission_id": app_notary_id or None,
            "stapled": mode == "formal",
            "staple_validated": mode == "formal",
            "gatekeeper_assessed": mode == "formal",
        },
        "dmg": {
            "status": notary_status,
            "submission_id": dmg_notary_id or None,
            "stapled": mode == "formal",
            "staple_validated": mode == "formal",
            "gatekeeper_assessed": mode == "formal",
        },
    },
    "validation": {
        "formal_gates_complete": formal_gates_complete,
        "app_signature": True,
        "siril_nested_signature": True,
        "siril_same_team": mode == "formal",
        "siril_python_entitlements": True,
        "native_runtime_manifest": True,
        "native_import_probe": True,
        "dmg_signature": True,
        "dmg_image_verify": True,
        "mounted_app_verified": True,
    },
}
canonical = json.dumps(
    manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
manifest["manifest_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
temporary = output_path.with_name(output_path.name + ".tmp")
temporary.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output_path)
print(
    "release_evidence="
    + ("eligible" if release_eligible else "blocked")
    + f" blockers={len(final_blockers)}"
)
PY
}

verify_release_sidecar() {
  local release_path="$1"
  local dmg_path="$2"
  local checksum_path="$3"
  local runtime_path="$4"

  "$VERIFY_PYTHON" - \
    "$release_path" "$dmg_path" "$checksum_path" "$runtime_path" "$MODE" \
    "$FORMAL_GATES_COMPLETE" "$VERIFIED_APP_TEAM" "$VERIFIED_APP_CDHASH" \
    "$VERIFIED_DMG_TEAM" "$VERIFIED_DMG_CDHASH" \
    "$APP_NOTARY_ID" "$DMG_NOTARY_ID" "$OUTPUT_NAME" \
    "$MOUNTED_RUNTIME_MANIFEST_SHA" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

(
    release_arg,
    dmg_arg,
    checksum_arg,
    runtime_arg,
    expected_mode,
    formal_gates_raw,
    expected_app_team,
    expected_app_cdhash,
    expected_dmg_team,
    expected_dmg_cdhash,
    expected_app_notary_id,
    expected_dmg_notary_id,
    expected_output_name,
    expected_runtime_sha,
) = sys.argv[1:]

release_path = Path(release_arg)
dmg_path = Path(dmg_arg)
checksum_path = Path(checksum_arg)
runtime_path = Path(runtime_arg)
for label, path in (
    ("release evidence", release_path),
    ("DMG", dmg_path),
    ("checksum", checksum_path),
    ("native runtime manifest", runtime_path),
):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"candidate {label} is missing, non-regular, or a symlink")
if dmg_path.name != expected_output_name:
    raise SystemExit("candidate DMG filename mismatch")
if checksum_path.name != expected_output_name + ".sha256":
    raise SystemExit("candidate checksum filename mismatch")
if release_path.name != expected_output_name + ".release.json":
    raise SystemExit("candidate release evidence filename mismatch")
if expected_mode not in {"local", "formal"} or formal_gates_raw not in {"0", "1"}:
    raise SystemExit("invalid expected packaging state")
formal_gates_complete = formal_gates_raw == "1"

def string_list(value, label):
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SystemExit(f"release evidence {label} is invalid")
    return value

def record(container, key, label):
    value = container.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"release evidence {label} is invalid")
    return value

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not re.fullmatch(r"[0-9a-f]{64}", expected_runtime_sha):
    raise SystemExit("expected mounted runtime manifest SHA-256 is invalid")
if sha256_file(runtime_path) != expected_runtime_sha:
    raise SystemExit("staged/mounted runtime manifest SHA-256 mismatch")
try:
    actual_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"native runtime manifest is unreadable: {error}")
if not isinstance(actual_runtime, dict):
    raise SystemExit("native runtime manifest root must be an object")
if actual_runtime.get("schema") != "starun.native-pipeline-runtime.v1":
    raise SystemExit("unexpected native runtime manifest schema")
actual_runtime_payload_sha = str(
    actual_runtime.get("manifest_payload_sha256") or ""
)
if not re.fullmatch(r"[0-9a-f]{64}", actual_runtime_payload_sha):
    raise SystemExit("native runtime manifest payload hash is invalid")
unsigned_runtime = dict(actual_runtime)
unsigned_runtime.pop("manifest_payload_sha256", None)
canonical_runtime = json.dumps(
    unsigned_runtime, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
if hashlib.sha256(canonical_runtime).hexdigest() != actual_runtime_payload_sha:
    raise SystemExit("native runtime manifest payload hash mismatch")
actual_runtime_blocking = string_list(
    actual_runtime.get("blocking_reasons"), "native runtime blocking_reasons"
)

try:
    manifest = json.loads(release_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"release evidence is unreadable: {error}")
if not isinstance(manifest, dict) or manifest.get("schema") != "starun.macos-release.v1":
    raise SystemExit("unexpected release evidence schema")
claimed = str(manifest.get("manifest_payload_sha256") or "")
unsigned = dict(manifest)
unsigned.pop("manifest_payload_sha256", None)
canonical = json.dumps(
    unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
if not re.fullmatch(r"[0-9a-f]{64}", claimed):
    raise SystemExit("release evidence payload hash is invalid")
if hashlib.sha256(canonical).hexdigest() != claimed:
    raise SystemExit("release evidence payload hash mismatch")
if manifest.get("mode") != expected_mode:
    raise SystemExit("release evidence mode mismatch")
if manifest.get("distribution_scope") != (
    "developer_id_notarized" if expected_mode == "formal" else "internal_only"
):
    raise SystemExit("release evidence distribution scope mismatch")
source_blocking = string_list(
    manifest.get("source_runtime_blocking_reasons"),
    "source_runtime_blocking_reasons",
)
if source_blocking != actual_runtime_blocking:
    raise SystemExit("release evidence changed native runtime blocking reasons")
resolved_blocking = string_list(
    manifest.get("resolved_blocking_reasons"), "resolved_blocking_reasons"
)
blocking = string_list(manifest.get("blocking_reasons"), "blocking_reasons")
eligible = manifest.get("release_eligible")
if not isinstance(eligible, bool):
    raise SystemExit("release evidence release_eligible must be boolean")
package_resolvable = {"app_notarization_missing", "dmg_notarization_missing"}
if expected_mode == "formal":
    expected_resolved = [
        reason for reason in source_blocking if reason in package_resolvable
    ]
    expected_blocking = [
        reason for reason in source_blocking if reason not in package_resolvable
    ]
    if not formal_gates_complete:
        raise SystemExit("formal release evidence lacks completed packaging gates")
    if resolved_blocking != expected_resolved or blocking != expected_blocking:
        raise SystemExit("formal release evidence changed runtime blocking reasons")
    if eligible is not (not blocking):
        raise SystemExit("formal release eligibility does not match remaining blockers")
else:
    expected_blocking = list(source_blocking)
    for reason in (
        "local_ad_hoc_distribution",
        "developer_id_signature_missing",
        "app_notarization_missing",
        "dmg_notarization_missing",
    ):
        if reason not in expected_blocking:
            expected_blocking.append(reason)
    if formal_gates_complete or resolved_blocking or blocking != expected_blocking:
        raise SystemExit("local release evidence has inconsistent blocking reasons")
    if eligible:
        raise SystemExit("local ad-hoc release evidence cannot be eligible")

artifacts = record(manifest, "artifacts", "artifacts")
app_record = record(artifacts, "app", "App artifact")
dmg_record = record(artifacts, "dmg", "DMG artifact")
checksum_record = record(artifacts, "checksum", "checksum artifact")
runtime_record = record(
    artifacts, "native_runtime_manifest", "native runtime manifest artifact"
)
if app_record.get("team_identifier") != (expected_app_team or None):
    raise SystemExit("release evidence App Team ID mismatch")
if str(app_record.get("cdhash") or "") != expected_app_cdhash.lower():
    raise SystemExit("release evidence App CDHash mismatch")
if dmg_record.get("team_identifier") != (expected_dmg_team or None):
    raise SystemExit("release evidence DMG Team ID mismatch")
if str(dmg_record.get("cdhash") or "") != expected_dmg_cdhash.lower():
    raise SystemExit("release evidence DMG CDHash mismatch")
for label, value in (
    ("App", expected_app_cdhash),
    ("DMG", expected_dmg_cdhash),
):
    if not re.fullmatch(r"[0-9A-Fa-f]{40,64}", value):
        raise SystemExit(f"expected {label} CDHash is invalid")
if dmg_record.get("filename") != expected_output_name:
    raise SystemExit("release evidence DMG filename mismatch")
if type(dmg_record.get("size_bytes")) is not int:
    raise SystemExit("release evidence DMG size is invalid")
if dmg_record["size_bytes"] != dmg_path.stat().st_size:
    raise SystemExit("release evidence DMG size mismatch")
if checksum_record.get("filename") != expected_output_name + ".sha256":
    raise SystemExit("release evidence checksum filename mismatch")
expected_sha = str(dmg_record.get("sha256") or "")
if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
    raise SystemExit("release evidence DMG SHA-256 is invalid")
digest = hashlib.sha256()
with dmg_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha:
    raise SystemExit("release evidence does not bind the candidate DMG")
try:
    checksum_text = checksum_path.read_text(encoding="ascii")
except (OSError, UnicodeError) as error:
    raise SystemExit(f"candidate checksum is unreadable: {error}")
if checksum_text != f"{expected_sha}  {expected_output_name}\n":
    raise SystemExit("candidate checksum does not exactly bind the candidate DMG")

bundle_name = str(app_record.get("bundle_name") or "")
expected_runtime_path = str(
    Path(bundle_name)
    / "Contents/Resources/pipeline/native-pipeline-runtime-manifest.json"
)
if not bundle_name or Path(bundle_name).name != bundle_name:
    raise SystemExit("release evidence App bundle name is invalid")
if runtime_record.get("path_in_app") != expected_runtime_path:
    raise SystemExit("release evidence native runtime manifest path mismatch")
if str(runtime_record.get("sha256") or "") != expected_runtime_sha:
    raise SystemExit("release evidence mounted runtime manifest SHA-256 mismatch")
if runtime_record.get("manifest_payload_sha256") != actual_runtime_payload_sha:
    raise SystemExit("release evidence runtime payload hash mismatch")
if runtime_record.get("schema") != actual_runtime.get("schema"):
    raise SystemExit("release evidence runtime manifest schema mismatch")

notarization = record(manifest, "notarization", "notarization")
expected_status = "Accepted" if expected_mode == "formal" else "not_submitted"
for kind, expected_id in (
    ("app", expected_app_notary_id),
    ("dmg", expected_dmg_notary_id),
):
    notary_record = record(notarization, kind, f"{kind} notarization")
    if notary_record.get("status") != expected_status:
        raise SystemExit(f"release evidence {kind} notarization status mismatch")
    if notary_record.get("submission_id") != (expected_id or None):
        raise SystemExit(f"release evidence {kind} notarization ID mismatch")
    for field in ("stapled", "staple_validated", "gatekeeper_assessed"):
        if notary_record.get(field) is formal_gates_complete:
            continue
        raise SystemExit(f"release evidence {kind} {field} state mismatch")

validation = record(manifest, "validation", "validation")
if validation.get("formal_gates_complete") is not formal_gates_complete:
    raise SystemExit("release evidence formal gate state mismatch")
for field in (
    "app_signature",
    "siril_nested_signature",
    "siril_python_entitlements",
    "native_runtime_manifest",
    "native_import_probe",
    "dmg_signature",
    "dmg_image_verify",
    "mounted_app_verified",
):
    if validation.get(field) is not True:
        raise SystemExit(f"release evidence validation flag is not true: {field}")
if validation.get("siril_same_team") is not (expected_mode == "formal"):
    raise SystemExit("release evidence Siril Team validation state mismatch")
print("release_evidence_verified=" + claimed)
PY
}

publish_outputs() {
  local candidate_dmg="$1"
  local candidate_sha="$2"
  local candidate_release="$3"
  local candidate_parent=""
  PUBLISH_BACKUP_DMG="$OUTPUT_PARENT/.$OUTPUT_NAME.previous.$$"
  PUBLISH_BACKUP_SHA="$OUTPUT_PARENT/.$OUTPUT_NAME.sha256.previous.$$"
  PUBLISH_BACKUP_RELEASE="$OUTPUT_PARENT/.$OUTPUT_NAME.release.json.previous.$$"
  PUBLISH_NEW_DMG=0
  PUBLISH_NEW_SHA=0
  PUBLISH_NEW_RELEASE=0

  [[ ! -e "$PUBLISH_BACKUP_DMG" && ! -L "$PUBLISH_BACKUP_DMG" ]] \
    || die "temporary DMG backup already exists: $PUBLISH_BACKUP_DMG"
  [[ ! -e "$PUBLISH_BACKUP_SHA" && ! -L "$PUBLISH_BACKUP_SHA" ]] \
    || die "temporary checksum backup already exists: $PUBLISH_BACKUP_SHA"
  [[ ! -e "$PUBLISH_BACKUP_RELEASE" && ! -L "$PUBLISH_BACKUP_RELEASE" ]] \
    || die "temporary release evidence backup already exists: $PUBLISH_BACKUP_RELEASE"

  require_file "$candidate_dmg" "Candidate DMG"
  require_file "$candidate_sha" "Candidate checksum"
  require_file "$candidate_release" "Candidate release evidence"
  [[ "$candidate_dmg" == "$BUILD_ROOT/$OUTPUT_NAME" ]] \
    || die "candidate DMG path does not match the transaction output name"
  [[ "$candidate_sha" == "$BUILD_ROOT/$OUTPUT_NAME.sha256" ]] \
    || die "candidate checksum path does not match the transaction output name"
  [[ "$candidate_release" == "$BUILD_ROOT/$OUTPUT_NAME.release.json" ]] \
    || die "candidate release evidence path does not match the transaction output name"
  candidate_parent="$(cd "$(dirname "$candidate_dmg")" && pwd -P)"
  [[ "$candidate_parent" == "$BUILD_ROOT" ]] \
    || die "candidate DMG is outside the staging directory: $candidate_dmg"
  [[ "$(cd "$(dirname "$candidate_sha")" && pwd -P)" == "$BUILD_ROOT" ]] \
    || die "candidate checksum is outside the staging directory: $candidate_sha"
  [[ "$(cd "$(dirname "$candidate_release")" && pwd -P)" == "$BUILD_ROOT" ]] \
    || die "candidate release evidence is outside the staging directory: $candidate_release"
  if ! (cd "$candidate_parent" \
      && /usr/bin/shasum -a 256 -c "$(basename "$candidate_sha")"); then
    die "candidate DMG/checksum verification failed before publication"
  fi
  verify_release_sidecar \
    "$candidate_release" \
    "$candidate_dmg" \
    "$candidate_sha" \
    "$DMG_ROOT/$APP_BASENAME/Contents/Resources/pipeline/native-pipeline-runtime-manifest.json"

  # The three flat output paths cannot be exchanged as one filesystem object.
  # Block handled signals during this short same-filesystem rename transaction;
  # EXIT cleanup rolls back any command failure before the commit flag clears.
  trap '' INT TERM HUP
  PUBLISH_ACTIVE=1

  if [[ -e "$OUTPUT_DMG" ]]; then
    /bin/mv "$OUTPUT_DMG" "$PUBLISH_BACKUP_DMG" \
      || die "failed to stage the previous DMG for atomic replacement"
  fi
  if [[ -e "$OUTPUT_SHA" ]]; then
    /bin/mv "$OUTPUT_SHA" "$PUBLISH_BACKUP_SHA" \
      || die "failed to stage the previous checksum for atomic replacement"
  fi
  if [[ -e "$OUTPUT_RELEASE" ]]; then
    /bin/mv "$OUTPUT_RELEASE" "$PUBLISH_BACKUP_RELEASE" \
      || die "failed to stage the previous release evidence for atomic replacement"
  fi

  if /bin/mv "$candidate_dmg" "$OUTPUT_DMG"; then
    PUBLISH_NEW_DMG=1
  fi
  if [[ "$PUBLISH_NEW_DMG" == "1" ]] && /bin/mv "$candidate_sha" "$OUTPUT_SHA"; then
    PUBLISH_NEW_SHA=1
  fi
  if [[ "$PUBLISH_NEW_DMG" == "1" && "$PUBLISH_NEW_SHA" == "1" ]] \
      && /bin/mv "$candidate_release" "$OUTPUT_RELEASE"; then
    PUBLISH_NEW_RELEASE=1
  fi

  if [[ "$PUBLISH_NEW_DMG" != "1" || "$PUBLISH_NEW_SHA" != "1" \
        || "$PUBLISH_NEW_RELEASE" != "1" ]]; then
    die "transactional DMG/checksum/release-evidence publication failed"
  fi

  PUBLISH_ACTIVE=0
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  if [[ -e "$PUBLISH_BACKUP_DMG" ]] && ! /bin/rm -f -- "$PUBLISH_BACKUP_DMG"; then
    echo "[DMG][WARN] Published successfully; stale prior DMG backup remains: $PUBLISH_BACKUP_DMG" >&2
  fi
  if [[ -e "$PUBLISH_BACKUP_SHA" ]] && ! /bin/rm -f -- "$PUBLISH_BACKUP_SHA"; then
    echo "[DMG][WARN] Published successfully; stale prior checksum backup remains: $PUBLISH_BACKUP_SHA" >&2
  fi
  if [[ -e "$PUBLISH_BACKUP_RELEASE" ]] \
      && ! /bin/rm -f -- "$PUBLISH_BACKUP_RELEASE"; then
    echo "[DMG][WARN] Published successfully; stale release evidence backup remains: $PUBLISH_BACKUP_RELEASE" >&2
  fi
}

LOCK_DIR="$OUTPUT_PARENT/.$OUTPUT_NAME.package.lock"
if ! /bin/mkdir "$LOCK_DIR"; then
  die "another packaging run is active or left a stale lock: $LOCK_DIR"
fi
LOCK_ACQUIRED=1

log "Mode: $MODE"
log "Input App: $APP_PATH"
log "Output DMG: $OUTPUT_DMG"

# Validate the immutable input before any packaging/notary side effects.
verify_app_bundle "$APP_PATH"

BUILD_ROOT="$(mktemp -d "$OUTPUT_PARENT/.starun_dmg_package.XXXXXX")"
STAGED_APP="$BUILD_ROOT/$APP_BASENAME"
DMG_ROOT="$BUILD_ROOT/dmg-root"
MOUNT_DIR="$BUILD_ROOT/mount"
CANDIDATE_DMG="$BUILD_ROOT/$OUTPUT_NAME"
CANDIDATE_SHA="$BUILD_ROOT/$OUTPUT_NAME.sha256"
CANDIDATE_RELEASE="$BUILD_ROOT/$OUTPUT_NAME.release.json"
APP_ZIP="$BUILD_ROOT/$APP_NAME-notary.zip"

/bin/mkdir -p "$DMG_ROOT" "$MOUNT_DIR"
/usr/bin/ditto "$APP_PATH" "$STAGED_APP"
verify_app_bundle "$STAGED_APP"

if [[ "$MODE" == "formal" ]]; then
  /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$STAGED_APP" "$APP_ZIP"
  submit_for_notarization "$APP_ZIP" "App ZIP" "$BUILD_ROOT/app-notary.json" app
  /usr/bin/xcrun stapler staple -v "$STAGED_APP"
  /usr/bin/xcrun stapler validate -v "$STAGED_APP"
  verify_app_bundle "$STAGED_APP"
  assess_app "$STAGED_APP"
fi

/bin/mv "$STAGED_APP" "$DMG_ROOT/$APP_BASENAME"
/bin/ln -s /Applications "$DMG_ROOT/Applications"

log "Creating compressed DMG..."
/usr/bin/hdiutil create \
  -volname "$VOLUME_NAME" \
  -srcfolder "$DMG_ROOT" \
  -format UDZO \
  -ov \
  "$CANDIDATE_DMG" >/dev/null

if [[ "$MODE" == "formal" ]]; then
  /usr/bin/codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$CANDIDATE_DMG"
else
  /usr/bin/codesign --force --sign - "$CANDIDATE_DMG"
fi
verify_dmg_signature "$CANDIDATE_DMG"

if [[ "$MODE" == "formal" ]]; then
  submit_for_notarization "$CANDIDATE_DMG" "DMG" "$BUILD_ROOT/dmg-notary.json" dmg
  /usr/bin/xcrun stapler staple -v "$CANDIDATE_DMG"
  /usr/bin/xcrun stapler validate -v "$CANDIDATE_DMG"
  verify_dmg_signature "$CANDIDATE_DMG"
  assess_dmg "$CANDIDATE_DMG"
fi

/usr/bin/hdiutil verify "$CANDIDATE_DMG" >/dev/null
log "Mounting final DMG read-only for post-package verification..."
# Mark the mount as cleanup-owned before attach so a signal cannot leave an
# attached image outside the EXIT trap's responsibility.
MOUNT_IMAGE="$CANDIDATE_DMG"
MOUNTED=1
ATTACH_PLIST="$BUILD_ROOT/attach.plist"
if ! /usr/bin/hdiutil attach \
    -readonly \
    -nobrowse \
    -mountpoint "$MOUNT_DIR" \
    -plist \
    "$CANDIDATE_DMG" >"$ATTACH_PLIST"; then
  die "failed to attach final DMG for verification"
fi
if ! MOUNT_DEVICE="$("$VERIFY_PYTHON" - "$ATTACH_PLIST" "$MOUNT_DIR" <<'PY'
import plistlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
mount_point = sys.argv[2]
try:
    with path.open("rb") as handle:
        payload = plistlib.load(handle)
except (OSError, plistlib.InvalidFileException) as error:
    raise SystemExit(f"invalid hdiutil attach plist: {error}")

for entity in payload.get("system-entities", []):
    if str(entity.get("mount-point") or "") == mount_point:
        device = str(entity.get("dev-entry") or "")
        if device.startswith("/dev/disk"):
            print(device)
            raise SystemExit(0)
raise SystemExit("hdiutil did not report the requested mount point/device")
PY
)"; then
  die "unable to identify the final DMG verification mount"
fi
[[ "$MOUNT_DEVICE" == /dev/disk* ]] \
  || die "unsafe DMG verification device: $MOUNT_DEVICE"
if ! mount_state pair; then
  die "final DMG is not mounted at the expected mount point/device"
fi

MOUNTED_APP="$MOUNT_DIR/$APP_BASENAME"
verify_app_bundle "$MOUNTED_APP"
MOUNTED_RUNTIME_MANIFEST="$MOUNTED_APP/Contents/Resources/pipeline/native-pipeline-runtime-manifest.json"
require_file "$MOUNTED_RUNTIME_MANIFEST" "Mounted native runtime manifest"
MOUNTED_RUNTIME_MANIFEST_SHA="$(/usr/bin/shasum -a 256 "$MOUNTED_RUNTIME_MANIFEST" \
  | /usr/bin/awk '{print $1}')"
[[ "$MOUNTED_RUNTIME_MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to calculate mounted runtime manifest SHA-256"
if [[ "$MODE" == "formal" ]]; then
  assess_app "$MOUNTED_APP"
fi
detach_mount || die "unable to detach the final DMG verification mount"
if [[ "$MODE" == "formal" ]]; then
  FORMAL_GATES_COMPLETE=1
fi

# This is deliberately after all formal stapling and mounted-payload checks.
FINAL_SHA="$(/usr/bin/shasum -a 256 "$CANDIDATE_DMG" | /usr/bin/awk '{print $1}')"
[[ "$FINAL_SHA" =~ ^[0-9a-f]{64}$ ]] || die "failed to calculate final DMG SHA-256"
printf '%s  %s\n' "$FINAL_SHA" "$OUTPUT_NAME" > "$CANDIDATE_SHA"
generate_release_sidecar \
  "$CANDIDATE_RELEASE" \
  "$CANDIDATE_DMG" \
  "$DMG_ROOT/$APP_BASENAME" \
  "$FINAL_SHA" \
  "$MOUNTED_RUNTIME_MANIFEST_SHA"

publish_outputs "$CANDIDATE_DMG" "$CANDIDATE_SHA" "$CANDIDATE_RELEASE"

log "Packaging complete."
log "DMG: $OUTPUT_DMG"
log "SHA-256: $FINAL_SHA"
log "Checksum: $OUTPUT_SHA"
log "Release evidence: $OUTPUT_RELEASE"
if [[ "$MODE" == "local" ]]; then
  log "Distribution scope: internal-only ad-hoc (not notarized)"
else
  log "Distribution scope: Developer ID, App and DMG notarized/stapled; release eligibility is recorded in the sidecar"
fi
