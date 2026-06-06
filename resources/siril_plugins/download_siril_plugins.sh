#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${ROOT_DIR}/downloads"
VENDOR_DIR="${ROOT_DIR}/vendor"
SYQON_DIR="${ROOT_DIR}/syqon_starless"
WHEEL_LOCK_FILE="${ROOT_DIR}/requirements-macos-arm64.lock"
ASSET_CHECKSUM_FILE="${ROOT_DIR}/asset-checksums.sha256"
TARGET_PYTHON_VERSION="313"
TARGET_ABI="cp313"
TARGET_PLATFORM="macosx_14_0_arm64"

SIRIL_ARCHIVE="${DOWNLOAD_DIR}/siril-scripts.tar.gz"
SIRIL_UNPACK_DIR="${VENDOR_DIR}/siril-scripts"

mkdir -p "${DOWNLOAD_DIR}" "${VENDOR_DIR}" "${SYQON_DIR}"

case "$(uname -m)" in
  arm64|aarch64)
    TARGET_PLATFORM="macosx_14_0_arm64"
    ;;
  *)
    echo "No SHA256 wheel lock is available for architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

expected_sha256() {
  local relative_path="$1"
  local checksum
  checksum="$(awk -v path="${relative_path}" '$2 == path { print $1; exit }' "${ASSET_CHECKSUM_FILE}")"
  if [[ ! "${checksum}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "Missing or invalid SHA256 for ${relative_path} in ${ASSET_CHECKSUM_FILE}" >&2
    exit 1
  fi
  printf '%s\n' "${checksum}"
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  local actual
  actual="$(shasum -a 256 "${file}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA256 verification failed for ${file}" >&2
    echo "  expected: ${expected}" >&2
    echo "  actual:   ${actual}" >&2
    return 1
  fi
}

download_verified() {
  local url="$1"
  local destination="$2"
  local relative_path="$3"
  local tmp="${destination}.tmp"
  local expected
  expected="$(expected_sha256 "${relative_path}")"
  rm -f "${tmp}"
  if ! curl -fL "${url}" -o "${tmp}"; then
    rm -f "${tmp}"
    echo "Download failed: ${url}" >&2
    exit 1
  fi
  if ! verify_sha256 "${tmp}" "${expected}"; then
    rm -f "${tmp}"
    exit 1
  fi
  mv "${tmp}" "${destination}"
}

if [[ ! -f "${ASSET_CHECKSUM_FILE}" || ! -f "${WHEEL_LOCK_FILE}" ]]; then
  echo "Integrity lock files are missing." >&2
  exit 1
fi

echo "[1/4] Downloading and verifying official siril-scripts archive..."
download_verified \
  "https://gitlab.com/free-astro/siril-scripts/-/archive/main/siril-scripts-main.tar.gz" \
  "${SIRIL_ARCHIVE}" \
  "downloads/siril-scripts.tar.gz"

echo "[2/4] Extracting siril-scripts..."
rm -rf "${SIRIL_UNPACK_DIR}.tmp"
mkdir -p "${SIRIL_UNPACK_DIR}.tmp"
tar -xzf "${SIRIL_ARCHIVE}" -C "${SIRIL_UNPACK_DIR}.tmp" --strip-components=1
rm -rf "${SIRIL_UNPACK_DIR}"
mv "${SIRIL_UNPACK_DIR}.tmp" "${SIRIL_UNPACK_DIR}"

echo "[3/4] Downloading hash-locked Python 3.13 wheels..."
python3 -m pip download \
  --no-deps \
  --require-hashes \
  --only-binary=:all: \
  --python-version "${TARGET_PYTHON_VERSION}" \
  --implementation cp \
  --abi "${TARGET_ABI}" \
  --abi abi3 \
  --platform "${TARGET_PLATFORM}" \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  -r "${WHEEL_LOCK_FILE}"

echo "[4/4] Downloading and verifying SyQon Starless offline cache..."
download_verified \
  "https://siril.syqon.it/syqon_starless_inference.py" \
  "${SYQON_DIR}/syqon_starless_inference.py" \
  "syqon_starless/syqon_starless_inference.py"
download_verified \
  "https://siril.syqon.it/zenith.pt.sha256" \
  "${SYQON_DIR}/zenith.pt.sha256" \
  "syqon_starless/zenith.pt.sha256"
download_verified \
  "https://siril.syqon.it/zenith.pt.date" \
  "${SYQON_DIR}/zenith.pt.date" \
  "syqon_starless/zenith.pt.date"
download_verified \
  "https://siril.syqon.it/zenith.pt" \
  "${SYQON_DIR}/zenith.pt" \
  "syqon_starless/zenith.pt"

upstream_zenith_sha="$(awk 'NR == 1 { print $1 }' "${SYQON_DIR}/zenith.pt.sha256")"
locked_zenith_sha="$(expected_sha256 "syqon_starless/zenith.pt")"
if [[ "${upstream_zenith_sha}" != "${locked_zenith_sha}" ]]; then
  echo "SyQon zenith.pt checksum file does not match the trusted lock." >&2
  exit 1
fi
verify_sha256 "${SYQON_DIR}/zenith.pt" "${upstream_zenith_sha}"

echo "Plugin download complete."
echo "Downloaded files:"
echo "  - ${SIRIL_ARCHIVE}"
echo "  - ${DOWNLOAD_DIR}/setiastrosuitepro-*.whl"
echo "  - ${DOWNLOAD_DIR}/onnxruntime-*.whl"
echo "  - ${DOWNLOAD_DIR}/pyqt6-*.whl"
echo "  - ${DOWNLOAD_DIR}/pyqt6_qt6-*.whl"
echo "  - ${DOWNLOAD_DIR}/pyqt6_sip-*.whl"
echo "  - ${DOWNLOAD_DIR}/pyside6-*.whl"
echo "  - ${DOWNLOAD_DIR}/pyside6_addons-*.whl"
echo "  - ${DOWNLOAD_DIR}/pyside6_essentials-*.whl"
echo "  - ${DOWNLOAD_DIR}/shiboken6-*.whl"
echo "  - ${DOWNLOAD_DIR}/torch-*.whl"
echo "  - ${DOWNLOAD_DIR}/torchvision-*.whl"
echo "  - ${DOWNLOAD_DIR}/sep-*.whl"
echo "  - ${DOWNLOAD_DIR}/spandrel-*.whl"
echo "  - ${DOWNLOAD_DIR}/lz4-*.whl"
echo "  - ${DOWNLOAD_DIR}/zstandard-*.whl"
echo "  - ${DOWNLOAD_DIR}/exifread-*.whl"
echo "  - ${DOWNLOAD_DIR}/opencv_python_headless-*.whl"
echo "  - ${DOWNLOAD_DIR}/requests-*.whl"
echo "  - ${DOWNLOAD_DIR}/setuptools-*.whl"
echo "  - ${DOWNLOAD_DIR}/wheel-*.whl"
echo "  - ${DOWNLOAD_DIR}/einops-*.whl"
echo "  - ${DOWNLOAD_DIR}/safetensors-*.whl"
echo "  - ${SYQON_DIR}/syqon_starless_inference.py"
echo "  - ${SYQON_DIR}/zenith.pt"
echo "  - ${DOWNLOAD_DIR}/astropy-*.whl"
echo "  - ${DOWNLOAD_DIR}/scipy-*.whl"
echo "  - ${DOWNLOAD_DIR}/tifffile-*.whl"
echo "Extracted scripts:"
echo "  - ${SIRIL_UNPACK_DIR}"
