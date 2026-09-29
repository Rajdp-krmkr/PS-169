"""
Scenario configuration and settings dialog for FSOC PAT Simulator.
"""

from __future__ import annotations
import json
import os
from typing import Optional, Dict, Any
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QFileDialog,
    QMessageBox,
)


class SettingsDialog(QDialog):
    """
    Dialog allowing configuration of simulation scenarios, camera optics, and controllers.
    """

    def __init__(self, current_config: Dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Simulation & Scenario Configuration")
        self.resize(500, 480)
        self.setStyleSheet("""
            QDialog { background-color: #0e121a; color: #cfd8dc; }
            QLabel { color: #b0bec5; font-size: 11px; }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #192030;
                color: #eceff1;
                border: 1px solid #2d3850;
                padding: 4px;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #1e2638;
                color: #ffffff;
                border: 1px solid #37474f;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #263238; border-color: #00e5ff; }
        """)

        self.config = json.loads(json.dumps(current_config))  # Deep copy

        layout = QVBoxLayout(self)

        # 1. Preset Selector
        preset_group = QGroupBox("PRESET SCENARIO")
        p_layout = QHBoxLayout(preset_group)
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(["DEFAULT", "EASY", "UAV", "SATELLITE", "EXTREME"])
        btn_apply_preset = QPushButton("LOAD PRESET")
        btn_apply_preset.clicked.connect(self._load_preset_file)
        p_layout.addWidget(self.combo_presets)
        p_layout.addWidget(btn_apply_preset)
        layout.addWidget(preset_group)

        # 2. Camera Optics
        cam_group = QGroupBox("CAMERA OPTICS & GIMBAL")
        form_cam = QFormLayout(cam_group)

        self.spin_fov_h = QDoubleSpinBox()
        self.spin_fov_h.setRange(2.0, 90.0)
        self.spin_fov_h.setValue(self.config.get("camera", {}).get("fov_horizontal_deg", 20.0))

        self.spin_fov_v = QDoubleSpinBox()
        self.spin_fov_v.setRange(2.0, 60.0)
        self.spin_fov_v.setValue(self.config.get("camera", {}).get("fov_vertical_deg", 11.25))

        self.spin_max_speed = QDoubleSpinBox()
        self.spin_max_speed.setRange(5.0, 100.0)
        self.spin_max_speed.setValue(self.config.get("camera", {}).get("max_pan_speed_deg_s", 25.0))

        form_cam.addRow("Horizontal FOV (deg):", self.spin_fov_h)
        form_cam.addRow("Vertical FOV (deg):", self.spin_fov_v)
        form_cam.addRow("Max Gimbal Speed (deg/s):", self.spin_max_speed)
        layout.addWidget(cam_group)

        # 3. Beacon & Trajectory
        b_group = QGroupBox("OPTICAL BEACON TARGET")
        form_b = QFormLayout(b_group)

        self.combo_traj = QComboBox()
        self.combo_traj.addItems(["sinusoidal", "linear", "circular", "maneuver"])
        current_traj = self.config.get("beacon", {}).get("trajectory", {}).get("type", "sinusoidal")
        self.combo_traj.setCurrentText(current_traj)

        self.spin_intensity = QSpinBox()
        self.spin_intensity.setRange(50, 255)
        self.spin_intensity.setValue(int(self.config.get("beacon", {}).get("intensity", 255)))

        form_b.addRow("Trajectory Type:", self.combo_traj)
        form_b.addRow("Beacon Peak Intensity:", self.spin_intensity)
        layout.addWidget(b_group)

        # 4. Action Buttons
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("IMPORT JSON...")
        btn_load.clicked.connect(self._import_json)
        btn_save = QPushButton("EXPORT JSON...")
        btn_save.clicked.connect(self._export_json)
        btn_ok = QPushButton("APPLY & CLOSE")
        btn_ok.clicked.connect(self._apply_and_close)

        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

    def _load_preset_file(self) -> None:
        preset_name = self.combo_presets.currentText().lower()
        path = os.path.join("configs", f"{preset_name}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            # Update fields
            cam = self.config.get("camera", {})
            self.spin_fov_h.setValue(cam.get("fov_horizontal_deg", 20.0))
            self.spin_fov_v.setValue(cam.get("fov_vertical_deg", 11.25))
            self.spin_max_speed.setValue(cam.get("max_pan_speed_deg_s", 25.0))
            traj = self.config.get("beacon", {}).get("trajectory", {}).get("type", "sinusoidal")
            self.combo_traj.setCurrentText(traj)
            QMessageBox.information(self, "Preset Loaded", f"Loaded '{preset_name.upper()}' preset.")

    def _import_json(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Import Config", "configs", "JSON Files (*.json)")
        if filename:
            with open(filename, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self._load_preset_file()

    def _export_json(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Export Config", "configs", "JSON Files (*.json)")
        if filename:
            self._sync_to_config()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

    def _sync_to_config(self) -> None:
        if "camera" not in self.config:
            self.config["camera"] = {}
        self.config["camera"]["fov_horizontal_deg"] = self.spin_fov_h.value()
        self.config["camera"]["fov_vertical_deg"] = self.spin_fov_v.value()
        self.config["camera"]["max_pan_speed_deg_s"] = self.spin_max_speed.value()
        self.config["camera"]["max_tilt_speed_deg_s"] = self.spin_max_speed.value()

        if "beacon" not in self.config:
            self.config["beacon"] = {}
        if "trajectory" not in self.config["beacon"]:
            self.config["beacon"]["trajectory"] = {}
        self.config["beacon"]["trajectory"]["type"] = self.combo_traj.currentText()
        self.config["beacon"]["intensity"] = self.spin_intensity.value()

    def _apply_and_close(self) -> None:
        self._sync_to_config()
        self.accept()
