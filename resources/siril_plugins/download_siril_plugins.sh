#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${ROOT_DIR}/downloads"
VENDOR_DIR="${ROOT_DIR}/vendor"
SYQON_DIR="${ROOT_DIR}/syqon_starless"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"
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
  x86_64|amd64)
    TARGET_PLATFORM="macosx_10_15_x86_64"
    ;;
  *)
    echo "Unsupported macOS architecture for wheel download: $(uname -m)" >&2
    exit 1
    ;;
esac

pip_download_cp313() {
  python3 -m pip download \
    --only-binary=:all: \
    --python-version "${TARGET_PYTHON_VERSION}" \
    --implementation cp \
    --abi "${TARGET_ABI}" \
    --abi abi3 \
    --platform "${TARGET_PLATFORM}" \
    --index-url "https://pypi.org/simple" \
    --dest "${DOWNLOAD_DIR}" \
    "$@"
}

prune_python_wheels() {
  python3 - "${DOWNLOAD_DIR}" "${TARGET_ABI}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

try:
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
except Exception as exc:
    raise SystemExit(f"Cannot import pip wheel parser: {exc}")

download_dir = Path(sys.argv[1])
target_abi = sys.argv[2]
allowed_interpreters = {target_abi, "py3"}
allowed_abis = {target_abi, "abi3", "none"}

removed: list[Path] = []
groups: dict[str, list[tuple[object, Path]]] = {}

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
            and tag.abi == "abi3"
        )
        for tag in tags
    )
    if not compatible:
        wheel.unlink()
        removed.append(wheel)
        continue

    groups.setdefault(canonicalize_name(name), []).append((version, wheel))

for _name, candidates in sorted(groups.items()):
    candidates.sort(key=lambda item: (item[0], item[1].name))
    if _name == "setuptools":
        compatible = [item for item in candidates if item[0].major < 82]
        if compatible:
            keep = compatible[-1]
        else:
            keep = candidates[-1]
    else:
        keep = candidates[-1]
    for _version, wheel in candidates:
        if wheel == keep[1]:
            continue
        wheel.unlink()
        removed.append(wheel)

print(f"Pruned Python wheels: removed={len(removed)}, kept={len(groups)}")
for wheel in removed:
    print(f"  removed {wheel.name}")
PY
}

echo "[1/8] Downloading official siril-scripts archive..."
SIRIL_URLS=(
  "https://gitlab.com/free-astro/siril-scripts/-/archive/main/siril-scripts-main.tar.gz"
  "https://gitlab.com/free-astro/siril-scripts/-/archive/master/siril-scripts-master.tar.gz"
)
download_ok=0
for url in "${SIRIL_URLS[@]}"; do
  if curl -fL "${url}" -o "${SIRIL_ARCHIVE}.tmp"; then
    mv "${SIRIL_ARCHIVE}.tmp" "${SIRIL_ARCHIVE}"
    download_ok=1
    break
  fi
done
if [[ "${download_ok}" -ne 1 ]]; then
  echo "Failed to download siril-scripts archive from known URLs." >&2
  exit 1
fi

echo "[2/8] Extracting siril-scripts..."
rm -rf "${SIRIL_UNPACK_DIR}.tmp"
mkdir -p "${SIRIL_UNPACK_DIR}.tmp"
tar -xzf "${SIRIL_ARCHIVE}" -C "${SIRIL_UNPACK_DIR}.tmp" --strip-components=1
rm -rf "${SIRIL_UNPACK_DIR}"
mv "${SIRIL_UNPACK_DIR}.tmp" "${SIRIL_UNPACK_DIR}"

echo "[3/8] Downloading SetiAstroSuitePro wheel (PyPI)..."
python3 -m pip download \
  --no-deps \
  --only-binary=:all: \
  --python-version "${TARGET_PYTHON_VERSION}" \
  --implementation py \
  --abi none \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  setiastrosuitepro

echo "[4/8] Downloading onnxruntime wheel for Python 3.13..."
pip_download_cp313 \
  --no-deps \
  onnxruntime

echo "[5/8] Downloading PyQt6/PySide6 wheels for Python 3.13..."
pip_download_cp313 \
  PyQt6

pip_download_cp313 \
  PySide6

echo "[6/8] Downloading astropy/scipy/tifffile/sep/spandrel/lz4/zstandard/exifread/opencv/requests/setuptools/wheel wheels for Python 3.13..."
pip_download_cp313 \
  astropy \
  scipy \
  tifffile \
  sep \
  spandrel \
  lz4 \
  zstandard \
  exifread \
  opencv-python-headless \
  requests \
  wheel

echo "[7/8] Downloading PyTorch wheels for SyQon Starless..."
pip_download_cp313 \
  torch \
  torchvision

echo "[7/8] Downloading requirement-declared Python 3.13 runtime wheels..."
pip_download_cp313 \
  -r "${REQUIREMENTS_FILE}"

echo "[7/8] Pruning duplicate/non-3.13 Python wheels..."
prune_python_wheels

echo "[8/8] Downloading SyQon Starless offline model cache..."
curl -fL "https://siril.syqon.it/syqon_starless_inference.py" \
  -o "${SYQON_DIR}/syqon_starless_inference.py"
curl -fL "https://siril.syqon.it/zenith.pt" \
  -o "${SYQON_DIR}/zenith.pt"
curl -fL "https://siril.syqon.it/zenith.pt.sha256" \
  -o "${SYQON_DIR}/zenith.pt.sha256"
curl -fL "https://siril.syqon.it/zenith.pt.date" \
  -o "${SYQON_DIR}/zenith.pt.date" || true

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
