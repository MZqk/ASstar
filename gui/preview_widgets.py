"""Focused image-preview widgets for the Starun macOS workspace."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView


class LatestPreviewCanvas(QGraphicsView):
    """Single-image canvas with fit, 1:1, wheel zoom, and hand panning."""

    MIN_SCALE = 0.05
    MAX_SCALE = 16.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("previewCanvas")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._fit_mode = True
        self._scale_factor = 1.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAccessibleName("当前阶段图像预览")
        self.setAccessibleDescription("显示 Stage 0 输入或最新完成阶段的原始预览")

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def set_image(self, path: Path | str) -> bool:
        # The pipeline atomically replaces one stable ``latest.png`` path.
        # QImageReader forces a fresh decode instead of reusing a cached pixmap.
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        pixmap = QPixmap.fromImage(reader.read())
        if pixmap.isNull():
            return False
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_window()
        return True

    def clear_image(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._fit_mode = True
        self._scale_factor = 1.0
        self.resetTransform()

    def fit_to_window(self) -> None:
        if self._pixmap_item is None:
            return
        self._fit_mode = True
        self.resetTransform()
        self.fitInView(
            self._pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._scale_factor = float(self.transform().m11())

    def actual_pixels(self) -> None:
        if self._pixmap_item is None:
            return
        self._fit_mode = False
        self.resetTransform()
        self._scale_factor = 1.0

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(0.8)

    def _zoom_by(self, factor: float) -> None:
        if self._pixmap_item is None:
            return
        target = self._scale_factor * float(factor)
        if target < self.MIN_SCALE or target > self.MAX_SCALE:
            return
        self._fit_mode = False
        self.scale(float(factor), float(factor))
        self._scale_factor = target

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._pixmap_item is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self._zoom_by(1.15 if delta > 0 else 1.0 / 1.15)
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()


__all__ = ["LatestPreviewCanvas"]
