"""
Reusable PySide6 custom widgets for FSOC PAT Mission-Control GUI.

Includes:
- MetricCard: High-contrast telemetry display box
- StateBannerWidget: Dynamic glowing state indicator
- SliderControl: Labeled precision slider with value readout
- DisturbanceControlPanel: Controls for all environmental and platform disturbances
- PIDControlPanel: Dynamic gains and dead-zone tuning controls
- ManualJogPanel: Manual gimbal steer controls
"""

from __future__ import annotations
from typing import Optional, Callable
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QPushButton,
    QGroupBox,
    QProgressBar,
    QDoubleSpinBox,
)


DARK_THEME_STYLE = """
QGroupBox {
    background-color: #131722;
    border: 1px solid #252b3b;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    font-size: 11px;
    color: #90a4ae;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #00e5ff;
}
QLabel {
    color: #cfd8dc;
    font-size: 11px;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #252b3b;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #00e5ff;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QPushButton {
    background-color: #1e2638;
    color: #eceff1;
    border: 1px solid #37474f;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
    font-size: 11px;
}
QPushButton:hover {
    background-color: #263238;
    border-color: #00e5ff;
}
QPushButton:pressed {
    background-color: #00b0ff;
    color: #000000;
}
"""


class MetricCard(QWidget):
    """Telemetry indicator card showing title, value, and unit."""

    def __init__(
        self,
        title: str,
        initial_value: str = "0.0",
        unit: str = "",
        accent_color: str = "#00e5ff",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.accent_color = accent_color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet("color: #78909c; font-size: 9px; font-weight: bold;")

        self.lbl_value = QLabel(initial_value)
        self.lbl_value.setStyleSheet(
            f"color: {accent_color}; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;"
        )

        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet("color: #546e7a; font-size: 9px;")

        val_row = QHBoxLayout()
        val_row.addWidget(self.lbl_value)
        val_row.addWidget(self.lbl_unit)
        val_row.addStretch()

        layout.addWidget(self.lbl_title)
        layout.addLayout(val_row)

        self.setStyleSheet(
            f"background-color: #10141e; border: 1px solid #1c2233; border-left: 3px solid {accent_color}; border-radius: 4px;"
        )

    def set_value(self, value_str: str, unit_str: Optional[str] = None) -> None:
        self.lbl_value.setText(value_str)
        if unit_str is not None:
            self.lbl_unit.setText(unit_str)
            # Dynamic color styling for states
            if unit_str == "READY":
                self.lbl_value.setStyleSheet(
                    "color: #00e676; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;"
                )
                self.lbl_unit.setStyleSheet("color: #00e676; font-size: 9px; font-weight: bold;")
            elif unit_str == "STABILIZING":
                self.lbl_value.setStyleSheet(
                    "color: #00e5ff; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;"
                )
                self.lbl_unit.setStyleSheet("color: #00e5ff; font-size: 9px; font-weight: bold;")
            elif unit_str == "NOT READY":
                self.lbl_value.setStyleSheet(
                    f"color: {self.accent_color}; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;"
                )
                self.lbl_unit.setStyleSheet("color: #ff5252; font-size: 9px; font-weight: bold;")

        # Dynamic color styling for Risk card values
        if value_str == "STABLE":
            self.lbl_value.setStyleSheet("color: #00e676; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;")
        elif value_str == "WARNING":
            self.lbl_value.setStyleSheet("color: #ffea00; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;")
        elif value_str in ("LOCK AT RISK", "CRITICAL"):
            self.lbl_value.setStyleSheet("color: #ff5252; font-size: 16px; font-weight: bold; font-family: 'Consolas', monospace;")


class StateBannerWidget(QWidget):
    """Prominent badge displaying the active PAT state with reactive coloring."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)

        lbl_prefix = QLabel("PAT STATUS:")
        lbl_prefix.setStyleSheet("color: #90a4ae; font-size: 11px; font-weight: bold;")

        self.lbl_state = QLabel("SEARCHING")
        self.lbl_state.setAlignment(Qt.AlignCenter)
        self.lbl_state.setStyleSheet(
            "background-color: #f57f17; color: #ffffff; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 12px; letter-spacing: 1px;"
        )

        layout.addWidget(lbl_prefix)
        layout.addWidget(self.lbl_state)
        layout.addStretch()

        self.setStyleSheet("background-color: #0d111a; border-radius: 6px; border: 1px solid #1e2638;")

    def set_state(self, state_str: str) -> None:
        state_upper = state_str.upper()
        self.lbl_state.setText(state_upper)

        color_map = {
            "TRACKING": "#00c853",    # Green
            "ACQUIRING": "#00b0ff",   # Blue
            "UNCERTAIN": "#ff9100",   # Orange
            "LOST": "#d50000",        # Red
            "REACQUIRING": "#aa00ff", # Purple
            "SEARCHING": "#f57f17",   # Amber
        }
        bg_color = color_map.get(state_upper, "#455a64")
        self.lbl_state.setStyleSheet(
            f"background-color: {bg_color}; color: #ffffff; padding: 4px 14px; border-radius: 4px; font-weight: bold; font-size: 12px; letter-spacing: 1px;"
        )


class SliderControl(QWidget):
    """Slider with precise float or integer scale and numeric readout."""

    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        min_val: float = 0.0,
        max_val: float = 100.0,
        initial_val: float = 0.0,
        unit: str = "%",
        is_float: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.is_float = is_float
        self.unit = unit

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.lbl_name = QLabel(label)
        self.lbl_name.setFixedWidth(110)
        self.lbl_name.setStyleSheet("color: #b0bec5; font-size: 11px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider_steps = 1000 if is_float else int(max_val - min_val)
        self.slider.setRange(0, self.slider_steps)

        self.lbl_val = QLabel("")
        self.lbl_val.setFixedWidth(55)
        self.lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_val.setStyleSheet("color: #00e5ff; font-family: monospace; font-weight: bold;")

        layout.addWidget(self.lbl_name)
        layout.addWidget(self.slider)
        layout.addWidget(self.lbl_val)

        self.set_value(initial_val)
        self.slider.valueChanged.connect(self._on_slider_moved)

    def _on_slider_moved(self, raw_val: int) -> None:
        val = self.get_value()
        if not self.is_float:
            self.lbl_val.setText(f"{int(val)} {self.unit}")
        else:
            self.lbl_val.setText(f"{val:.2f} {self.unit}".strip())
        self.value_changed.emit(val)

    def get_value(self) -> float:
        frac = self.slider.value() / float(max(1, self.slider_steps))
        val = self.min_val + frac * (self.max_val - self.min_val)
        return float(round(val) if not self.is_float else val)

    def set_value(self, val: float) -> None:
        clamped = max(self.min_val, min(self.max_val, val))
        frac = (clamped - self.min_val) / max(1e-5, self.max_val - self.min_val)
        self.slider.setValue(int(round(frac * self.slider_steps)))
        if not self.is_float:
            self.lbl_val.setText(f"{int(clamped)} {self.unit}")
        else:
            self.lbl_val.setText(f"{clamped:.2f} {self.unit}".strip())


class DisturbanceControlPanel(QGroupBox):
    """Panel grouping all disturbance sliders with quick presets."""

    disturbances_changed = Signal(dict)
    occlusion_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("DISTURBANCE ENGINE (0-100%)", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.slider_noise = SliderControl("Sensor Noise", 0, 100, 0, "%")
        self.slider_vib = SliderControl("Vibration", 0, 100, 0, "%")
        self.slider_turb = SliderControl("Turbulence", 0, 100, 0, "%")
        self.slider_blur = SliderControl("Motion Blur", 0, 100, 0, "%")
        self.slider_defocus = SliderControl("Defocus", 0, 100, 0, "%")

        layout.addWidget(self.slider_noise)
        layout.addWidget(self.slider_vib)
        layout.addWidget(self.slider_turb)
        layout.addWidget(self.slider_blur)
        layout.addWidget(self.slider_defocus)

        # Occlusion button
        self.btn_occlude = QPushButton("TEMPORARILY OCCLUDE TARGET")
        self.btn_occlude.setCheckable(True)
        self.btn_occlude.setStyleSheet(
            "QPushButton:checked { background-color: #d50000; color: white; border-color: #ff1744; }"
        )
        self.btn_occlude.toggled.connect(self._on_occlude_toggled)
        layout.addWidget(self.btn_occlude)

        # Quick preset buttons
        preset_row = QHBoxLayout()
        btn_zero = QPushButton("ZERO")
        btn_zero.clicked.connect(lambda: self.set_preset(0, 0, 0, 0, 0))
        btn_uav = QPushButton("UAV")
        btn_uav.clicked.connect(lambda: self.set_preset(12, 15, 15, 5, 0))
        btn_extreme = QPushButton("EXTREME")
        btn_extreme.clicked.connect(lambda: self.set_preset(25, 30, 25, 20, 15))

        preset_row.addWidget(btn_zero)
        preset_row.addWidget(btn_uav)
        preset_row.addWidget(btn_extreme)
        layout.addLayout(preset_row)

        for s in [self.slider_noise, self.slider_vib, self.slider_turb, self.slider_blur, self.slider_defocus]:
            s.value_changed.connect(self._emit_change)

    def _emit_change(self) -> None:
        self.disturbances_changed.emit({
            "noise": self.slider_noise.get_value(),
            "vibration": self.slider_vib.get_value(),
            "turbulence": self.slider_turb.get_value(),
            "motion_blur": self.slider_blur.get_value(),
            "defocus": self.slider_defocus.get_value(),
        })

    def _on_occlude_toggled(self, checked: bool) -> None:
        self.btn_occlude.setText("TARGET OCCLUDED (CLICK TO RESTORE)" if checked else "TEMPORARILY OCCLUDE TARGET")
        self.occlusion_toggled.emit(checked)

    def set_preset(self, noise: float, vib: float, turb: float, blur: float, defocus: float) -> None:
        self.slider_noise.set_value(noise)
        self.slider_vib.set_value(vib)
        self.slider_turb.set_value(turb)
        self.slider_blur.set_value(blur)
        self.slider_defocus.set_value(defocus)
        self._emit_change()


class PIDControlPanel(QGroupBox):
    """Panel for real-time PID tuning and dead-zone modification."""

    pid_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("CLOSED-LOOP CONTROLLER TUNING", parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self.slider_kp = SliderControl("Kp (Proportional)", 0.0, 5.0, 1.8, "")
        self.slider_ki = SliderControl("Ki (Integral)", 0.0, 0.5, 0.05, "")
        self.slider_kd = SliderControl("Kd (Derivative)", 0.0, 1.0, 0.18, "")
        self.slider_dz = SliderControl("Dead-Zone", 0.0, 0.20, 0.04, "deg")
        self.slider_pred = SliderControl("Lookahead Pred", 0, 15, 5, "steps", is_float=False)

        layout.addWidget(self.slider_kp)
        layout.addWidget(self.slider_ki)
        layout.addWidget(self.slider_kd)
        layout.addWidget(self.slider_dz)
        layout.addWidget(self.slider_pred)

        for s in [self.slider_kp, self.slider_ki, self.slider_kd, self.slider_dz, self.slider_pred]:
            s.value_changed.connect(self._emit_change)

    def _emit_change(self) -> None:
        self.pid_changed.emit({
            "kp": self.slider_kp.get_value(),
            "ki": self.slider_ki.get_value(),
            "kd": self.slider_kd.get_value(),
            "dead_zone_deg": self.slider_dz.get_value(),
            "prediction_steps": int(self.slider_pred.get_value()),
        })


class ManualJogPanel(QGroupBox):
    """Manual directional gimbal jog buttons for operator override."""

    jog_command = Signal(float, float)  # pan_rate, tilt_rate

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("MANUAL GIMBAL JOG OVERRIDE", parent)
        grid = QGridLayout(self)
        grid.setSpacing(4)

        btn_up = QPushButton("▲ UP")
        btn_down = QPushButton("▼ DOWN")
        btn_left = QPushButton("◀ LEFT")
        btn_right = QPushButton("▶ RIGHT")
        btn_stop = QPushButton("■ STOP")

        rate = 8.0  # deg/s
        btn_up.clicked.connect(lambda: self.jog_command.emit(0.0, rate))
        btn_down.clicked.connect(lambda: self.jog_command.emit(0.0, -rate))
        btn_left.clicked.connect(lambda: self.jog_command.emit(-rate, 0.0))
        btn_right.clicked.connect(lambda: self.jog_command.emit(rate, 0.0))
        btn_stop.clicked.connect(lambda: self.jog_command.emit(0.0, 0.0))

        grid.addWidget(btn_up, 0, 1)
        grid.addWidget(btn_left, 1, 0)
        grid.addWidget(btn_stop, 1, 1)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_down, 2, 1)
