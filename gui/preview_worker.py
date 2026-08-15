"""Asynchronous Stage 0 FITS preview generation for the macOS GUI."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QThread, Signal

try:
    from pipeline.ui_preview import write_display_fits_preview
except ImportError:  # Support direct execution from the gui directory.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.ui_preview import write_display_fits_preview  # type: ignore[no-redef]


def preview_cache_path(work_dir: Path) -> Path:
    """Return a deterministic cache path without writing into the user's project."""
    digest = hashlib.sha256(str(work_dir).encode("utf-8")).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / "starun-previews" / digest
    return root / "stage0_input.png"


class InitialPreviewWorker(QThread):
    """Read the first valid Stage 0 FITS candidate off the UI thread."""

    ready = Signal(int, str, str)
    failed = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        candidates: Iterable[Path],
        output_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.request_id = int(request_id)
        self.candidates = tuple(Path(path) for path in candidates)
        self.output_path = Path(output_path)

    def run(self) -> None:
        try:
            source = write_display_fits_preview(
                self.candidates,
                self.output_path,
                apply_stretch=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.failed.emit(self.request_id, str(exc))
        else:
            self.ready.emit(self.request_id, str(self.output_path), str(source))


__all__ = ["InitialPreviewWorker", "preview_cache_path"]
