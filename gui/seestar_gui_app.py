#!/usr/bin/env python3
"""Seestar Superimpose macOS GUI launcher entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .main_window import (
        QApplication,
        DiskSpaceEstimate,
        INPUT_MODE_AUTO,
        INPUT_MODE_LINEAR_RESUME,
        INPUT_MODE_STAGE2_CORRECTED_RESUME,
        LINEAR_RESUME_INPUT_NAME,
        STAGE2_CORRECTED_INPUT_NAME,
        PipelineWorker,
        SeestarGui,
        build_siril_cli_command,
        default_runtime_home,
        resource_root,
        resolve_siril_scripts_root,
        resolve_venv_site_packages,
        siril_state_root_from_home,
    )
except ImportError:  # Support direct execution: python gui/seestar_gui_app.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from main_window import (  # type: ignore[no-redef]
        QApplication,
        DiskSpaceEstimate,
        INPUT_MODE_AUTO,
        INPUT_MODE_LINEAR_RESUME,
        INPUT_MODE_STAGE2_CORRECTED_RESUME,
        LINEAR_RESUME_INPUT_NAME,
        STAGE2_CORRECTED_INPUT_NAME,
        PipelineWorker,
        SeestarGui,
        build_siril_cli_command,
        default_runtime_home,
        resource_root,
        resolve_siril_scripts_root,
        resolve_venv_site_packages,
        siril_state_root_from_home,
    )


def main() -> int:
    app = QApplication(sys.argv)
    window = SeestarGui()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
