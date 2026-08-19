#!/usr/bin/env python3
"""Starun macOS GUI launcher entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer

try:
    from .main_window import (
        QApplication,
        BootstrapWorker,
        DiskSpaceEstimate,
        INPUT_MODE_AUTO,
        INPUT_MODE_LINEAR_RESUME,
        INPUT_MODE_STAGE2_CORRECTED_RESUME,
        STAGE2_CORRECTED_INPUT_NAME,
        PipelineWorker,
        StarunGui,
        build_siril_cli_command,
        default_runtime_home,
        is_siril_cp312_wheel_compatible,
        resource_root,
        resolve_siril_scripts_root,
        resolve_venv_site_packages,
        siril_state_root_from_home,
    )
except ImportError:  # Support direct execution: python gui/starun_gui_app.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from main_window import (  # type: ignore[no-redef]
        QApplication,
        BootstrapWorker,
        DiskSpaceEstimate,
        INPUT_MODE_AUTO,
        INPUT_MODE_LINEAR_RESUME,
        INPUT_MODE_STAGE2_CORRECTED_RESUME,
        STAGE2_CORRECTED_INPUT_NAME,
        PipelineWorker,
        StarunGui,
        build_siril_cli_command,
        default_runtime_home,
        is_siril_cp312_wheel_compatible,
        resource_root,
        resolve_siril_scripts_root,
        resolve_venv_site_packages,
        siril_state_root_from_home,
    )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Starun")
    app.setApplicationDisplayName("Starun")
    app.setOrganizationName("Starun")
    app.setQuitOnLastWindowClosed(True)
    try:
        from .accessibility_safety import (
            install_macos_accessibility_selection_guard,
        )
    except ImportError:  # Support direct execution from the gui directory.
        from accessibility_safety import (  # type: ignore[no-redef]
            install_macos_accessibility_selection_guard,
        )

    install_macos_accessibility_selection_guard()
    try:
        from .ui_theme import install_application_theme
    except ImportError:
        from ui_theme import install_application_theme  # type: ignore[no-redef]
    install_application_theme(app)
    window = StarunGui()
    app.screenRemoved.connect(window._handle_screen_removed)
    window.show()
    QTimer.singleShot(0, window._show_main_window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
