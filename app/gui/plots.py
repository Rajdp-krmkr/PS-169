"""
Real-time high-performance telemetry strip-charts using PyQtGraph.

Plots:
1. Pointing Error vs Time (deg) with dead-zone lock threshold
2. Camera Attitude (Pan & Tilt angles vs Time in deg)
3. Target Position vs Boresight Reticle (Phase Plane)
4. System Frame Rate (FPS vs Time)
"""

from __future__ import annotations
import collections
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QHBoxLayout
import pyqtgraph as pg
import numpy as np


class RealtimeTelemetryPlots(QWidget):
    """
    High-refresh PyQtGraph telemetry streaming suite.
    """

    def __init__(self, history_len: int = 180, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.history_len = history_len

        # Configure PyQtGraph global options for dark aerospace HUD styling
        pg.setConfigOption("background", "#0d111a")
        pg.setConfigOption("foreground", "#90a4ae")
        pg.setConfigOption("antialias", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Tabbed container for clean organization
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1c2436; border-radius: 4px; background: #0d111a; }
            QTabBar::tab { background: #131a26; color: #90a4ae; padding: 4px 10px; font-size: 10px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1e293b; color: #00e5ff; border-bottom: 2px solid #00e5ff; }
        """)

        # Data buffers
        self.time_buf = collections.deque(maxlen=history_len)
        self.err_buf = collections.deque(maxlen=history_len)
        self.pan_buf = collections.deque(maxlen=history_len)
        self.tilt_buf = collections.deque(maxlen=history_len)
        self.fps_buf = collections.deque(maxlen=history_len)
        self.target_x_buf = collections.deque(maxlen=history_len)
        self.target_y_buf = collections.deque(maxlen=history_len)

        # --- TAB 1: Tracking Error & Attitude ---
        tab1_widget = QWidget()
        tab1_layout = QHBoxLayout(tab1_widget)
        tab1_layout.setContentsMargins(2, 2, 2, 2)

        # Plot A: Pointing Error
        self.plot_err = pg.PlotWidget(title="ANGULAR POINTING ERROR (deg)")
        self.plot_err.showGrid(x=True, y=True, alpha=0.25)
        self.plot_err.setLabel("left", "Error", units="deg")
        self.plot_err.setLabel("bottom", "Time", units="s")
        self.curve_err = self.plot_err.plot(pen=pg.mkPen(color="#00e5ff", width=2))

        # Dead-zone threshold line
        self.thresh_line = pg.InfiniteLine(
            pos=0.04, angle=0, pen=pg.mkPen(color="#00e676", width=1.5, style=pg.QtCore.Qt.DashLine)
        )
        self.plot_err.addItem(self.thresh_line)

        # Plot B: Pan & Tilt Attitude
        self.plot_att = pg.PlotWidget(title="GIMBAL ATTITUDE (deg)")
        self.plot_att.showGrid(x=True, y=True, alpha=0.25)
        self.plot_att.setLabel("left", "Angle", units="deg")
        self.plot_att.setLabel("bottom", "Time", units="s")
        self.plot_att.addLegend(offset=(10, 10))
        self.curve_pan = self.plot_att.plot(name="Pan (Az)", pen=pg.mkPen(color="#00e5ff", width=1.8))
        self.curve_tilt = self.plot_att.plot(name="Tilt (El)", pen=pg.mkPen(color="#ff4081", width=1.8))

        tab1_layout.addWidget(self.plot_err)
        tab1_layout.addWidget(self.plot_att)
        self.tabs.addTab(tab1_widget, "TRACKING & ATTITUDE")

        # --- TAB 2: Phase Plane & FPS ---
        tab2_widget = QWidget()
        tab2_layout = QHBoxLayout(tab2_widget)
        tab2_layout.setContentsMargins(2, 2, 2, 2)

        # Plot C: Boresight Phase Plane (dx vs dy)
        self.plot_phase = pg.PlotWidget(title="BORESIGHT PHASE TRAJECTORY (deg)")
        self.plot_phase.showGrid(x=True, y=True, alpha=0.25)
        self.plot_phase.setLabel("left", "Delta Tilt", units="deg")
        self.plot_phase.setLabel("bottom", "Delta Pan", units="deg")
        self.curve_phase = self.plot_phase.plot(
            pen=pg.mkPen(color="#ffea00", width=1.5),
            symbol="o",
            symbolSize=4,
            symbolBrush="#ffea00",
        )
        # Center target marker
        center_mark = pg.ScatterPlotItem(
            pos=[(0, 0)], size=10, pen=pg.mkPen("#00e5ff"), brush=pg.mkBrush("#00e5ff")
        )
        self.plot_phase.addItem(center_mark)

        # Plot D: FPS Telemetry
        self.plot_fps = pg.PlotWidget(title="SIMULATION FPS")
        self.plot_fps.showGrid(x=True, y=True, alpha=0.25)
        self.plot_fps.setLabel("left", "Rate", units="FPS")
        self.plot_fps.setLabel("bottom", "Time", units="s")
        self.curve_fps = self.plot_fps.plot(pen=pg.mkPen(color="#69f0ae", width=1.8))

        tab2_layout.addWidget(self.plot_phase)
        tab2_layout.addWidget(self.plot_fps)
        self.tabs.addTab(tab2_widget, "PHASE PLANE & FPS")

        layout.addWidget(self.tabs)

    def append_telemetry(
        self,
        t_sec: float,
        error_deg: float,
        pan_deg: float,
        tilt_deg: float,
        delta_az_deg: float,
        delta_el_deg: float,
        fps: float,
    ) -> None:
        """Append data points to ring buffers and update plots safely."""
        try:
            self.time_buf.append(float(t_sec))
            self.err_buf.append(float(error_deg))
            self.pan_buf.append(float(pan_deg))
            self.tilt_buf.append(float(tilt_deg))
            self.target_x_buf.append(float(delta_az_deg))
            self.target_y_buf.append(float(delta_el_deg))
            self.fps_buf.append(float(fps))

            if len(self.time_buf) < 2:
                return

            t_arr = np.array(self.time_buf, dtype=np.float64)
            self.curve_err.setData(t_arr, np.array(self.err_buf, dtype=np.float64))
            self.curve_pan.setData(t_arr, np.array(self.pan_buf, dtype=np.float64))
            self.curve_tilt.setData(t_arr, np.array(self.tilt_buf, dtype=np.float64))
            self.curve_phase.setData(
                np.array(self.target_x_buf, dtype=np.float64),
                np.array(self.target_y_buf, dtype=np.float64),
            )
            self.curve_fps.setData(t_arr, np.array(self.fps_buf, dtype=np.float64))
        except Exception:
            pass  # Suppress plot exceptions to prevent GUI crashes

    def set_deadzone_threshold(self, deadzone_deg: float) -> None:
        """Update horizontal reference line on error plot safely."""
        try:
            self.thresh_line.setValue(float(deadzone_deg))
        except Exception:
            pass

    def reset_buffers(self) -> None:
        """Clear plot history safely."""
        try:
            self.time_buf.clear()
            self.err_buf.clear()
            self.pan_buf.clear()
            self.tilt_buf.clear()
            self.fps_buf.clear()
            self.target_x_buf.clear()
            self.target_y_buf.clear()
            self.curve_err.clear()
            self.curve_pan.clear()
            self.curve_tilt.clear()
            self.curve_phase.clear()
            self.curve_fps.clear()
        except Exception:
            pass
