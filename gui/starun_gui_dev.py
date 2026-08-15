#!/usr/bin/env python3
"""Run the source GUI against an installed Siril app and an existing seed."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_RESOURCES = PROJECT_ROOT / "resources"
DEFAULT_SIRIL_APP = Path("/Applications/Siril.app")
DEFAULT_SIRIL_SEED = (
    Path.home() / "Library/Application Support/org.siril.Siril/siril"
)


class DevLauncherError(RuntimeError):
    """Raised when the explicit development runtime is incomplete."""


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def validate_siril_app(siril_app: Path) -> None:
    required_executables = (
        siril_app / "Contents/MacOS/siril-cli",
        siril_app
        / "Contents/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
    )
    if not siril_app.is_dir():
        raise DevLauncherError(f"Siril App 不存在：{siril_app}")
    for path in required_executables:
        if not path.is_file():
            raise DevLauncherError(f"Siril 运行文件缺失：{path}")
        if not path.stat().st_mode & 0o111:
            raise DevLauncherError(f"Siril 运行文件不可执行：{path}")


def validate_siril_seed(siril_seed: Path) -> None:
    required_paths = (
        siril_seed / "venv",
        siril_seed / ".python_module/sirilpy",
    )
    if not siril_seed.is_dir():
        raise DevLauncherError(f"Siril seed 不存在：{siril_seed}")
    for path in required_paths:
        if not path.is_dir():
            raise DevLauncherError(f"Siril seed 不完整，缺少目录：{path}")


def prepare_resource_overlay(
    overlay_root: Path,
    *,
    project_resources: Path,
    siril_app: Path,
    siril_seed: Path,
) -> Path:
    """Create a temporary development resource tree without copying resources."""
    validate_siril_app(siril_app)
    validate_siril_seed(siril_seed)
    if not project_resources.is_dir():
        raise DevLauncherError(f"项目资源目录不存在：{project_resources}")

    overlay_root.mkdir(parents=True, exist_ok=True)
    reserved_names = {"Siril.app", "SirilPythonSeed"}
    for source in project_resources.iterdir():
        if source.name in reserved_names:
            continue
        target = overlay_root / source.name
        target.symlink_to(
            source.resolve(),
            target_is_directory=source.is_dir(),
        )

    # Mirror the frozen App layout so the GUI resolves the pipeline from the
    # injected runtime resource tree instead of reaching back into the checkout.
    project_pipeline = project_resources.parent / "pipeline"
    if project_pipeline.is_dir():
        (overlay_root / "pipeline").symlink_to(
            project_pipeline.resolve(),
            target_is_directory=True,
        )

    (overlay_root / "Siril.app").symlink_to(
        siril_app.resolve(),
        target_is_directory=True,
    )
    (overlay_root / "SirilPythonSeed").symlink_to(
        siril_seed.resolve(),
        target_is_directory=True,
    )
    return overlay_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "不打包运行 Starun 源码 GUI，并显式使用系统 Siril 与现有 "
            "Siril Python seed。"
        )
    )
    parser.add_argument(
        "--siril-app",
        type=_absolute_path,
        default=DEFAULT_SIRIL_APP,
        help=f"Siril.app 路径（默认：{DEFAULT_SIRIL_APP}）",
    )
    parser.add_argument(
        "--siril-seed",
        type=_absolute_path,
        default=DEFAULT_SIRIL_SEED,
        help=f"包含 venv 与 .python_module 的 seed 根目录（默认：{DEFAULT_SIRIL_SEED}）",
    )
    parser.add_argument(
        "--runtime-home",
        type=_absolute_path,
        default=None,
        help=(
            "可选的 Starun runtime HOME；不传时继续使用 "
            "~/Library/Application Support/Starun/runtime_home"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_siril_app(args.siril_app)
        validate_siril_seed(args.siril_seed)
    except DevLauncherError as error:
        parser.error(str(error))

    with tempfile.TemporaryDirectory(prefix="starun_gui_dev_resources_") as td:
        overlay = prepare_resource_overlay(
            Path(td),
            project_resources=PROJECT_RESOURCES,
            siril_app=args.siril_app,
            siril_seed=args.siril_seed,
        )

        try:
            from .main_window import QApplication, StarunGui
        except ImportError:  # Direct execution: python gui/starun_gui_dev.py
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from main_window import QApplication, StarunGui  # type: ignore[no-redef]
        from PySide6.QtCore import QTimer

        app = QApplication([sys.argv[0]])
        app.setApplicationName("Starun")
        app.setApplicationDisplayName("Starun")
        app.setOrganizationName("Starun")
        app.setQuitOnLastWindowClosed(True)
        try:
            from .ui_theme import install_application_theme
        except ImportError:
            from ui_theme import install_application_theme  # type: ignore[no-redef]
        install_application_theme(app)
        window = StarunGui(
            resources_override=overlay,
            runtime_home_override=args.runtime_home,
        )
        app.screenRemoved.connect(window._handle_screen_removed)
        window.show()
        QTimer.singleShot(0, window._show_main_window)
        return app.exec()


if __name__ == "__main__":
    sys.exit(main())
