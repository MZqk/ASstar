from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


SIRIL_REQUIRED_SITE_PACKAGES = ("sirilpy", "numpy", "packaging", "requests")
SIRIL_VENDOR_FALLBACK_PACKAGES = (
    "packaging",
    "requests",
    "urllib3",
    "charset_normalizer",
    "idna",
    "certifi",
)


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

    probe_code = "import sirilpy, numpy, packaging, requests; print('sirilpy-ok')"
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
