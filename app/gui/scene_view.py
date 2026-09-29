"""
Interactive Camera Viewport Widget with aspect-ratio preservation and HUD rendering.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox


class CameraSceneView(QWidget):
    """
    Renders the live camera optical viewport with HUD graphics.
    """

    manual_jog_drag = Signal(float, float)  # relative drag delta

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setStyleSheet("background-color: #080a0f; border-radius: 6px; border: 1px solid #1a2030;")

        self.pixmap: Optional[QPixmap] = None
        self.raw_frame_size = (1280, 720)
        self.last_mouse_pos: Optional[QPoint] = None

        # HUD Layer toggles
        self.show_gt = True
        self.show_pred = True
        self.show_reticle = True
        self.show_overview = True

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Receive new frame from simulation thread and convert to QPixmap safely."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        h, w, ch = frame_bgr.shape
        self.raw_frame_size = (w, h)

        # Ensure contiguous memory buffer
        if not frame_bgr.flags['C_CONTIGUOUS']:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        # Convert BGR OpenCV image to RGB QImage safely with independent memory copy
        bytes_per_line = ch * w
        q_img = QImage(
            frame_bgr.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_BGR888,
        ).copy()
        self.pixmap = QPixmap.fromImage(q_img)
        self.update()  # Trigger repaint

    def paintEvent(self, event) -> None:
        """Draw pixmap centered with aspect ratio scaling safely."""
        if self.width() <= 0 or self.height() <= 0:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            if self.pixmap is not None and not self.pixmap.isNull():
                scaled = self.pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                if not scaled.isNull():
                    x = (self.width() - scaled.width()) // 2
                    y = (self.height() - scaled.height()) // 2
                    painter.drawPixmap(x, y, scaled)
            else:
                painter.setPen(QColor("#546e7a"))
                painter.setFont(QFont("Consolas", 12))
                painter.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    "AWAITING OPTICAL SENSOR FEED...\nCLICK 'PLAY' TO INITIATE COARSE PAT",
                )
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        if self.last_mouse_pos is not None:
            pos = event.pos()
            dx = pos.x() - self.last_mouse_pos.x()
            dy = pos.y() - self.last_mouse_pos.y()
            scale = 0.05  # deg/s per pixel drag
            self.manual_jog_drag.emit(dx * scale, -dy * scale)
            self.last_mouse_pos = pos

    def mouseReleaseEvent(self, event) -> None:
        self.last_mouse_pos = None
        self.manual_jog_drag.emit(0.0, 0.0)  # Stop jog

    def leaveEvent(self, event) -> None:
        if self.last_mouse_pos is not None:
            self.last_mouse_pos = None
            self.manual_jog_drag.emit(0.0, 0.0)  # Stop jog when cursor leaves viewport
