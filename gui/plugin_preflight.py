from __future__ import annotations

from pathlib import Path


SYQON_STARLESS_BUNDLE_REL = Path("syqon_starless")
COSMIC_CLARITY_BUNDLE_REL = Path("cosmic_clarity")
COSMIC_CLARITY_REQUIRED_MODEL_FILES = (
    "deep_denoise_mono_AI4.pth",
    "deep_denoise_color_AI4.pth",
    "deep_sharp_stellar_AI4.pth",
    "deep_nonstellar_sharp_conditional_psf_AI4.pth",
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
