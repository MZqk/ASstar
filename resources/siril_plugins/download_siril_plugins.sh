#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="${ROOT_DIR}/downloads"
VENDOR_DIR="${ROOT_DIR}/vendor"
SYQON_DIR="${ROOT_DIR}/syqon_starless"

SIRIL_ARCHIVE="${DOWNLOAD_DIR}/siril-scripts.tar.gz"
SIRIL_UNPACK_DIR="${VENDOR_DIR}/siril-scripts"

mkdir -p "${DOWNLOAD_DIR}" "${VENDOR_DIR}" "${SYQON_DIR}"

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
  --python-version 312 \
  --implementation py \
  --abi none \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  setiastrosuitepro

echo "[4/8] Downloading onnxruntime wheel for Siril Python 3.12..."
ONNX_PLATFORM="macosx_11_0_arm64"
case "$(uname -m)" in
  arm64|aarch64)
    ONNX_PLATFORM="macosx_11_0_arm64"
    ;;
  x86_64|amd64)
    ONNX_PLATFORM="macosx_10_15_x86_64"
    ;;
esac
python3 -m pip download \
  --no-deps \
  --only-binary=:all: \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  --platform "${ONNX_PLATFORM}" \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  onnxruntime

echo "[5/8] Downloading PyQt6/PySide6 wheels for Siril Python 3.12..."
python3 -m pip download \
  --only-binary=:all: \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  --abi abi3 \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  PyQt6

python3 -m pip download \
  --only-binary=:all: \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  --abi abi3 \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  PySide6

echo "[6/8] Downloading astropy/scipy/tifffile/sep/spandrel/lz4/zstandard/exifread/opencv/requests/wheel wheels for Siril Python 3.12..."
python3 -m pip download \
  --only-binary=:all: \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  --abi abi3 \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
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
TORCH_PLATFORM="macosx_11_0_arm64"
case "$(uname -m)" in
  arm64|aarch64)
    TORCH_PLATFORM="macosx_11_0_arm64"
    ;;
  x86_64|amd64)
    TORCH_PLATFORM="macosx_10_15_x86_64"
    ;;
esac
python3 -m pip download \
  --only-binary=:all: \
  --python-version 312 \
  --implementation cp \
  --abi cp312 \
  --platform "${TORCH_PLATFORM}" \
  --index-url "https://pypi.org/simple" \
  --dest "${DOWNLOAD_DIR}" \
  torch \
  torchvision

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
