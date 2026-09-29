"""
Mission-Control GUI Dashboard for Free Space Optical Communication (FSOC) Coarse PAT.

Features:
- Sleek dark aerospace HUD aesthetics with responsive real-time layout
- Multithreaded simulation worker decoupled from Qt GUI thread (60 FPS UI)
- Live optical camera sensor viewport with HUD overlay and tactical starfield map
- PyQtGraph real-time strip-charts (Error, Pan/Tilt Attitude, Phase Trajectory, FPS)
- Interactive disturbance sliders (Noise, Vibration, Turbulence, Blur) with instant feedback
- Dynamic PID gains and dead-zone tuning controls
- Quick scenario preset selector (EASY, UAV, SATELLITE, EXTREME)
- Manual gimbal jog controls via buttons or viewport drag
"""

from __future__ import annotations
import math
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer, Slot, QMutex, QMutexLocker
from PySide6.QtGui import QIcon, QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSplitter,
    QTabWidget,
    QToolBar,
    QComboBox,
    QPushButton,
    QLabel,
    QCheckBox,
    QStatusBar,
    QMessageBox,
)

# Internal imports
from simulation.coordinate_system import rad2deg, deg2rad
from simulation.environment import build_environment_from_config, VirtualEnvironment
from simulation.camera import VirtualCamera
from simulation.hud import render_hud_overlay
from vision.classical_detector import ClassicalBeaconDetector
from vision.yolo_detector import YOLODetector
from vision.fusion import DetectionFusion
from control.camera_controller import CameraGimbalController
from control.search_strategy import SpiralScan, RasterScan
from tracking.tracker import PATTracker
from tracking.state_machine import PATState
from disturbances.manager import DisturbanceManager

from app.gui.scene_view import CameraSceneView
from app.gui.plots import RealtimeTelemetryPlots
from app.gui.widgets import (
    MetricCard,
    StateBannerWidget,
    DisturbanceControlPanel,
    PIDControlPanel,
    ManualJogPanel,
    DARK_THEME_STYLE,
)
from app.gui.settings import SettingsDialog


class SimulationWorker(QThread):
    """
    Background simulation worker thread running the closed-loop PAT dynamics.
    Decoupled from GUI thread with thread-safe mutex locking to prevent frame lag or UI crashing.
    """

    frame_ready = Signal(np.ndarray, dict)
    telemetry_updated = Signal(dict)
    state_changed = Signal(str)

    def __init__(self, config: Dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.running = True
        self.paused = False
        self.step_one = False
        self.mutex = QMutex()

        self._pending_config: Optional[Dict[str, Any]] = None
        self._pending_reset: bool = False

        self.fps = config.get("camera", {}).get("fps", 30.0)
        self.dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index = 0

        # Build simulation subsystems
        self._init_subsystems(config)

    def _init_subsystems(self, config: Dict[str, Any]) -> None:
        """Instantiate environment, camera, detector, tracker, and controllers."""
        self.env, self.camera = build_environment_from_config(config)

        det_cfg = config.get("detection", {})
        self.detector_mode = config.get("detector_mode", "hybrid").lower()
        self.detector_cv = ClassicalBeaconDetector(
            binary_threshold=det_cfg.get("binary_threshold", 150),
            min_area=det_cfg.get("min_area", 4.0),
            max_area=det_cfg.get("max_area", 1200.0),
            min_circularity=det_cfg.get("min_circularity", 0.50),
        )
        self.detector_ai = YOLODetector(model_path=config.get("ai_model_path", "models/beacon_detector.onnx"))
        self.fusion = DetectionFusion()

        track_cfg = config.get("tracking", {})
        self.tracker = PATTracker(
            camera=self.camera,
            prediction_horizon_steps=track_cfg.get("prediction_horizon_steps", 5),
            search_type=track_cfg.get("search_strategy", "spiral"),
        )

        pid_cfg = config.get("pid", {})
        self.controller = CameraGimbalController(
            camera=self.camera,
            kp_pan=pid_cfg.get("kp_pan", 2.0),
            ki_pan=pid_cfg.get("ki_pan", 0.08),
            kd_pan=pid_cfg.get("kd_pan", 0.18),
            kp_tilt=pid_cfg.get("kp_tilt", 2.0),
            ki_tilt=pid_cfg.get("ki_tilt", 0.08),
            kd_tilt=pid_cfg.get("kd_tilt", 0.18),
            dead_zone_deg=pid_cfg.get("dead_zone_deg", 0.04),
        )

        dist_cfg = config.get("disturbances", {})
        self.dist_mgr = DisturbanceManager(resolution=self.camera.resolution)
        self.dist_mgr.set_strengths(
            noise=dist_cfg.get("sensor_noise_percent", 0.0),
            vibration=dist_cfg.get("vibration_percent", 0.0),
            turbulence=dist_cfg.get("turbulence_percent", 0.0),
            motion_blur=dist_cfg.get("motion_blur_percent", 0.0),
        )

        self.primary_beacon = self.env.get_primary_beacon()
        self.manual_jog_rates = (0.0, 0.0)

    def run(self) -> None:
        """Main simulation thread execution loop."""
        while self.running:
            # 0. Check and safely process pending configuration or reset requests
            with QMutexLocker(self.mutex):
                if self._pending_config is not None:
                    self.config = self._pending_config
                    self._pending_config = None
                    self.sim_time = 0.0
                    self.frame_index = 0
                    self._init_subsystems(self.config)
                elif self._pending_reset:
                    self._pending_reset = False
                    self.sim_time = 0.0
                    self.frame_index = 0
                    self._init_subsystems(self.config)

            if self.paused and not self.step_one:
                self.msleep(30)
                continue

            self.step_one = False
            t_start = time.perf_counter()

            try:
                with QMutexLocker(self.mutex):
                    # 1. Physics update
                    self.env.update(self.sim_time, self.dt)

                    # 2. Optical disturbances (platform vibration injected into LOS)
                    self.dist_mgr.step_optical(self.camera, self.sim_time)

                    # 3. Viewport rendering
                    bg = self.env.generate_camera_background(self.camera)
                    raw_frame = self.camera.render_frame(self.env.beacons, self.sim_time, base_canvas=bg)

                    # 4. Visual distortions (turbulence, motion blur, noise, frame drop)
                    frame, dropped = self.dist_mgr.apply_visual(raw_frame, self.camera, self.sim_time)

                    # 5. Vision detection
                    if not dropped and not self.dist_mgr.occlusion_active:
                        if self.detector_mode == "classical":
                            det_res = self.detector_cv.detect(frame, timestamp=self.sim_time)
                        elif self.detector_mode == "ai":
                            det_res = self.detector_ai.detect(frame, timestamp=self.sim_time)
                        else:
                            cv_res = self.detector_cv.detect(frame, timestamp=self.sim_time)
                            ai_res = self.detector_ai.detect(frame, timestamp=self.sim_time)
                            det_res = self.fusion.fuse(cv_res, ai_res, timestamp=self.sim_time)
                        detected = det_res.primary
                        det_latency = det_res.latency_ms
                    else:
                        detected = None
                        det_latency = 0.0

                    # 6. PAT Tracker update
                    tracker_out = self.tracker.process_frame(detected, sim_time=self.sim_time, dt=self.dt)
                    if tracker_out.state_changed:
                        self.state_changed.emit(tracker_out.state.value)

                    # 7. Closed-loop camera control
                    if self.manual_jog_rates != (0.0, 0.0):
                        # Manual operator override
                        self.camera.set_command_rate(self.manual_jog_rates[0], self.manual_jog_rates[1])
                    elif tracker_out.search_rate_rad_s is not None:
                        # Acquisition scan
                        self.camera.set_command_rate(
                            tracker_out.search_rate_rad_s[0], tracker_out.search_rate_rad_s[1]
                        )
                    else:
                        # Closed-loop PID tracking
                        self.controller.track_angular_error(
                            tracker_out.cmd_delta_az_rad, tracker_out.cmd_delta_el_rad, self.dt
                        )

                    # 8. Gimbal step
                    self.camera.step(self.dt)

                    # 9. Ground Truth evaluation
                    if self.primary_beacon:
                        gt_az, gt_el, gt_u, gt_v, in_fov = self.camera.project_target(
                            self.primary_beacon.pos[0], self.primary_beacon.pos[1]
                        )
                    else:
                        gt_az, gt_el, gt_u, gt_v, in_fov = 0.0, 0.0, 0.0, 0.0, False

                    ang_err_deg = rad2deg(math.hypot(gt_az, gt_el))
                    is_locked = bool(in_fov and (ang_err_deg <= rad2deg(self.controller.pid_pan.dead_zone) * 2.5))

                    elapsed = time.perf_counter() - t_start
                    step_fps = 1.0 / max(1e-5, elapsed)

                    # 10. HUD overlay synthesis
                    hud_frame = render_hud_overlay(
                        frame=frame,
                        detection=detected,
                        ground_truth_uv=(gt_u, gt_v) if in_fov else None,
                        camera_pan_deg=rad2deg(self.camera.effective_pan),
                        camera_tilt_deg=rad2deg(self.camera.effective_tilt),
                        angular_error_deg=ang_err_deg,
                        dead_zone_pixels=self.camera.projection.rad_per_pixel_x * self.controller.dead_zone_rad,
                        fps=step_fps,
                        state_str=tracker_out.state.value,
                        is_locked=is_locked,
                        predicted_uv=tracker_out.predicted_pixel_uv,
                        safe_fov_box_pixels=tracker_out.safe_fov_box_pixels,
                        risk_state=tracker_out.risk_state,
                        handoff_score_pct=tracker_out.handoff_score_pct,
                        handoff_state=tracker_out.handoff_state,
                        active_filter=tracker_out.active_filter,
                    )

                    # Inset tactical world map
                    overview = self.env.render_overview_map(self.camera, map_size=(290, 168))
                    oh, ow = overview.shape[:2]
                    hud_frame[48 : 48 + oh, hud_frame.shape[1] - ow - 10 : hud_frame.shape[1] - 10] = overview

                    telemetry = {
                        "sim_time": self.sim_time,
                        "frame_idx": self.frame_index,
                        "state": tracker_out.state.value,
                        "error_deg": ang_err_deg,
                        "pan_deg": rad2deg(self.camera.effective_pan),
                        "tilt_deg": rad2deg(self.camera.effective_tilt),
                        "delta_az_deg": rad2deg(gt_az),
                        "delta_el_deg": rad2deg(gt_el),
                        "est_vel_pan": rad2deg(tracker_out.estimated_vel_rad_s[0]),
                        "est_vel_tilt": rad2deg(tracker_out.estimated_vel_rad_s[1]),
                        "fps": step_fps,
                        "is_locked": is_locked,
                        "det_latency_ms": det_latency,
                        "handoff_score_pct": tracker_out.handoff_score_pct,
                        "handoff_state": tracker_out.handoff_state,
                        "risk_state": tracker_out.risk_state,
                        "risk_score": tracker_out.risk_score,
                        "active_filter": tracker_out.active_filter,
                    }

                self.frame_ready.emit(hud_frame, telemetry)
                self.telemetry_updated.emit(telemetry)

                self.sim_time += self.dt
                self.frame_index += 1

            except Exception as e:
                print(f"[WARN] Simulation worker step error: {e}")

            # Throttle loop to match target FPS pacing
            sleep_needed = self.dt - (time.perf_counter() - t_start)
            if sleep_needed > 0.001:
                self.msleep(int(sleep_needed * 1000.0))

    def reset_simulation(self) -> None:
        """Request thread-safe simulation reset."""
        with QMutexLocker(self.mutex):
            self._pending_reset = True

    def set_config(self, config: Dict[str, Any]) -> None:
        """Request thread-safe reinitialization with new scenario configuration."""
        with QMutexLocker(self.mutex):
            self._pending_config = config


class FSOCMissionControlDashboard(QMainWindow):
    """
    Main Aerospace Mission-Control Window for FSOC PAT tracking.
    """

    def __init__(self, initial_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.setWindowTitle("SIH 2026 — FSOC Coarse PAT Virtual Camera Tracking System")
        self.resize(1440, 860)
        self.setStyleSheet(DARK_THEME_STYLE)

        # Default configuration fallback
        if initial_config is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "uav.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                self.config = {"scenario_name": "UAV"}
        else:
            self.config = initial_config

        # Cumulative metrics tracking
        self.errors_history: List[float] = []
        self.locked_frames_count = 0
        self.total_frames_count = 0

        self._build_ui()
        self._init_worker()

    def _build_ui(self) -> None:
        """Assemble all widgets, toolbars, viewports, and docks."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(6)

        # 1. Top Control Toolbar
        self._build_toolbar()

        # 2. Main Vertical Splitter: Top (Viewport + Docks) / Bottom (Plots)
        v_splitter = QSplitter(Qt.Vertical)

        # --- Top Section (Horizontal Splitter) ---
        h_splitter = QSplitter(Qt.Horizontal)

        # Left: Viewport & State Banner
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.state_banner = StateBannerWidget()
        self.camera_view = CameraSceneView()
        self.camera_view.manual_jog_drag.connect(self._on_manual_jog_drag)

        left_layout.addWidget(self.state_banner)
        left_layout.addWidget(self.camera_view, stretch=1)
        h_splitter.addWidget(left_box)

        # Right: Tabbed Control & Telemetry Deck
        right_deck = QTabWidget()
        right_deck.setFixedWidth(420)
        right_deck.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e2638; background: #0f131c; border-radius: 4px; }
            QTabBar::tab { background: #141a26; color: #90a4ae; padding: 6px 12px; font-weight: bold; font-size: 11px; }
            QTabBar::tab:selected { background: #1c2436; color: #00e5ff; border-bottom: 2px solid #00e5ff; }
        """)

        # Tab 1: Live Status & Metrics
        tab_status = QWidget()
        status_layout = QVBoxLayout(tab_status)
        status_layout.setSpacing(6)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(6)
        self.card_err = MetricCard("Pointing Error", "0.000", "deg", "#00e5ff")
        self.card_rmse = MetricCard("RMSE", "0.000", "deg", "#00e5ff")
        self.card_pan = MetricCard("Camera Pan", "+00.00", "deg", "#40c4ff")
        self.card_tilt = MetricCard("Camera Tilt", "+00.00", "deg", "#ff4081")
        self.card_vel = MetricCard("Target Est Speed", "0.00", "deg/s", "#ffea00")
        self.card_fps = MetricCard("System FPS", "00.0", "FPS", "#00e676")
        self.card_lock = MetricCard("Lock Retention", "0.0", "%", "#00e676")
        self.card_lat = MetricCard("Detector Latency", "0.00", "ms", "#ffd600")
        self.card_handoff = MetricCard("Fine-PAT Handoff", "0.0%", "NOT READY", "#00e676")
        self.card_risk = MetricCard("Loss-of-Lock Risk", "STABLE", "Score: 0.10", "#ff9100")

        metrics_grid.addWidget(self.card_err, 0, 0)
        metrics_grid.addWidget(self.card_rmse, 0, 1)
        metrics_grid.addWidget(self.card_pan, 1, 0)
        metrics_grid.addWidget(self.card_tilt, 1, 1)
        metrics_grid.addWidget(self.card_vel, 2, 0)
        metrics_grid.addWidget(self.card_fps, 2, 1)
        metrics_grid.addWidget(self.card_lock, 3, 0)
        metrics_grid.addWidget(self.card_lat, 3, 1)
        metrics_grid.addWidget(self.card_handoff, 4, 0)
        metrics_grid.addWidget(self.card_risk, 4, 1)

        status_layout.addLayout(metrics_grid)
        status_layout.addStretch()
        right_deck.addTab(tab_status, "TELEMETRY")

        # Tab 2: Disturbance Sliders
        tab_dist = QWidget()
        dist_layout = QVBoxLayout(tab_dist)
        self.panel_dist = DisturbanceControlPanel()
        self.panel_dist.disturbances_changed.connect(self._on_disturbances_changed)
        self.panel_dist.occlusion_toggled.connect(self._on_occlusion_toggled)
        dist_layout.addWidget(self.panel_dist)
        dist_layout.addStretch()
        right_deck.addTab(tab_dist, "DISTURBANCES")

        # Tab 3: Controller Tuning & Manual Jog
        tab_ctrl = QWidget()
        ctrl_layout = QVBoxLayout(tab_ctrl)
        self.panel_pid = PIDControlPanel()
        self.panel_pid.pid_changed.connect(self._on_pid_changed)
        self.panel_jog = ManualJogPanel()
        self.panel_jog.jog_command.connect(self._on_manual_jog_buttons)
        ctrl_layout.addWidget(self.panel_pid)
        ctrl_layout.addWidget(self.panel_jog)
        ctrl_layout.addStretch()
        right_deck.addTab(tab_ctrl, "CONTROL")

        h_splitter.addWidget(right_deck)
        h_splitter.setStretchFactor(0, 3)
        h_splitter.setStretchFactor(1, 1)

        v_splitter.addWidget(h_splitter)

        # --- Bottom Section (PyQtGraph Real-Time Plots) ---
        self.plots = RealtimeTelemetryPlots(history_len=200)
        self.plots.setFixedHeight(220)
        v_splitter.addWidget(self.plots)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(v_splitter)

        # Status Bar
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("background-color: #0a0d14; color: #78909c; font-size: 10px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("SYSTEM READY. Standalone Optical Gimbal PAT Loop Active.")

    def _build_toolbar(self) -> None:
        """Create sleek top mission-control toolbar."""
        toolbar = QToolBar("Simulation Controls", self)
        toolbar.setStyleSheet("""
            QToolBar { background-color: #10141e; border-bottom: 1px solid #1e2638; spacing: 8px; padding: 4px; }
            QLabel { color: #90a4ae; font-size: 11px; font-weight: bold; }
            QComboBox { background-color: #192030; color: #eceff1; border: 1px solid #2d3850; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        """)
        self.addToolBar(toolbar)

        # Brand Title
        lbl_brand = QLabel(" FSOC COARSE PAT ")
        lbl_brand.setStyleSheet("color: #00e5ff; font-size: 13px; font-weight: bold; letter-spacing: 1.5px;")
        toolbar.addWidget(lbl_brand)
        toolbar.addSeparator()

        # Play / Pause / Step / Reset
        self.btn_play = QPushButton("▶ PLAY")
        self.btn_play.clicked.connect(self._toggle_pause)
        self.btn_step = QPushButton("⏭ STEP")
        self.btn_step.clicked.connect(self._step_single_frame)
        self.btn_reset = QPushButton("↺ RESET")
        self.btn_reset.clicked.connect(self._reset_simulation)

        toolbar.addWidget(self.btn_play)
        toolbar.addWidget(self.btn_step)
        toolbar.addWidget(self.btn_reset)
        toolbar.addSeparator()

        # Scenario Preset Selector
        toolbar.addWidget(QLabel(" SCENARIO: "))
        self.combo_scenario = QComboBox()
        self.combo_scenario.addItems(["UAV", "EASY", "SATELLITE", "EXTREME", "DEFAULT"])
        self.combo_scenario.currentTextChanged.connect(self._on_scenario_selected)
        toolbar.addWidget(self.combo_scenario)
        toolbar.addSeparator()

        # Detector Mode Selector
        toolbar.addWidget(QLabel(" DETECTOR: "))
        self.combo_detector = QComboBox()
        self.combo_detector.addItems(["Hybrid Fusion", "Classical CV", "AI YOLO"])
        self.combo_detector.currentTextChanged.connect(self._on_detector_mode_changed)
        toolbar.addWidget(self.combo_detector)
        toolbar.addSeparator()

        # Advanced Settings & Report
        btn_settings = QPushButton("⚙ CONFIG")
        btn_settings.clicked.connect(self._open_settings_dialog)
        toolbar.addWidget(btn_settings)

    def _init_worker(self) -> None:
        """Start the background simulation worker thread."""
        self.worker = SimulationWorker(self.config)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.telemetry_updated.connect(self._on_telemetry_updated)
        self.worker.state_changed.connect(self.state_banner.set_state)
        self.worker.start()

    # --- Worker Signal Handlers ---
    @Slot(np.ndarray, dict)
    def _on_frame_ready(self, frame: np.ndarray, telemetry: dict) -> None:
        self.camera_view.update_frame(frame)

    @Slot(dict)
    def _on_telemetry_updated(self, telem: dict) -> None:
        try:
            # Update metric cards
            err = telem["error_deg"]
            self.errors_history.append(err)
            self.total_frames_count += 1
            if telem["is_locked"]:
                self.locked_frames_count += 1

            rmse = float(np.sqrt(np.mean(np.square(self.errors_history[-100:]))))
            lock_rate = (self.locked_frames_count / max(1, self.total_frames_count)) * 100.0

            self.card_err.set_value(f"{err:05.3f}")
            self.card_rmse.set_value(f"{rmse:05.3f}")
            self.card_pan.set_value(f"{telem['pan_deg']:+06.2f}")
            self.card_tilt.set_value(f"{telem['tilt_deg']:+06.2f}")
            est_speed = math.hypot(telem["est_vel_pan"], telem["est_vel_tilt"])
            self.card_vel.set_value(f"{est_speed:04.2f}")
            self.card_fps.set_value(f"{telem['fps']:4.1f}")
            self.card_lock.set_value(f"{lock_rate:4.1f}")
            self.card_lat.set_value(f"{telem['det_latency_ms']:04.2f}")
            self.card_handoff.set_value(
                f"{telem.get('handoff_score_pct', 0.0):.1f}%",
                telem.get("handoff_state", "NOT READY")
            )
            self.card_risk.set_value(
                telem.get("risk_state", "STABLE"),
                f"Score: {telem.get('risk_score', 0.10):.2f}"
            )

            # Update PyQtGraph strip charts
            if self.total_frames_count % 3 == 0:
                self.plots.append_telemetry(
                    t_sec=telem["sim_time"],
                    error_deg=err,
                    pan_deg=telem["pan_deg"],
                    tilt_deg=telem["tilt_deg"],
                    delta_az_deg=telem["delta_az_deg"],
                    delta_el_deg=telem["delta_el_deg"],
                    fps=telem["fps"],
                )

            self.status_bar.showMessage(
                f"SIM: {telem['sim_time']:.2f}s | PAT STATE: {telem['state']} | "
                f"LOCK: {'YES' if telem['is_locked'] else 'ACQUIRING'} | "
                f"ERROR: {err:.3f} deg | FRAMES: {self.total_frames_count}"
            )
        except Exception as e:
            pass

    # --- UI Interactions ---
    def _toggle_pause(self) -> None:
        self.worker.paused = not self.worker.paused
        self.btn_play.setText("▶ PLAY" if self.worker.paused else "⏸ PAUSE")

    def _step_single_frame(self) -> None:
        self.worker.paused = True
        self.worker.step_one = True
        self.btn_play.setText("▶ PLAY")

    def _reset_simulation(self) -> None:
        self.errors_history.clear()
        self.locked_frames_count = 0
        self.total_frames_count = 0
        self.plots.reset_buffers()
        self.worker.reset_simulation()
        self.status_bar.showMessage("Simulation state reset.")

    def _on_scenario_selected(self, preset_name: str) -> None:
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs",
            f"{preset_name.lower()}.json",
        )
        if os.path.exists(cfg_path):
            import json
            with open(cfg_path, "r", encoding="utf-8") as f:
                new_cfg = json.load(f)
            self.config = new_cfg
            self.errors_history.clear()
            self.locked_frames_count = 0
            self.total_frames_count = 0
            self.plots.reset_buffers()
            self.worker.set_config(new_cfg)
            self.status_bar.showMessage(f"Loaded scenario preset '{preset_name}'.")

    def _on_detector_mode_changed(self, mode_text: str) -> None:
        mode_map = {
            "Hybrid Fusion": "hybrid",
            "Classical CV": "classical",
            "AI YOLO": "ai",
        }
        with QMutexLocker(self.worker.mutex):
            self.worker.detector_mode = mode_map.get(mode_text, "hybrid")
        self.status_bar.showMessage(f"Detector mode set to: {self.worker.detector_mode.upper()}")

    def _on_disturbances_changed(self, d: dict) -> None:
        with QMutexLocker(self.worker.mutex):
            self.worker.dist_mgr.set_strengths(
                noise=d["noise"],
                vibration=d["vibration"],
                turbulence=d["turbulence"],
                motion_blur=d["motion_blur"],
                defocus=d["defocus"],
            )

    def _on_occlusion_toggled(self, occluded: bool) -> None:
        with QMutexLocker(self.worker.mutex):
            self.worker.dist_mgr.occlusion_active = occluded
        if occluded:
            self.status_bar.showMessage("[ALERT] Optical target occluded manually (testing reacquisition).")
        else:
            self.status_bar.showMessage("[INFO] Target optical occlusion lifted.")

    def _on_pid_changed(self, p: dict) -> None:
        with QMutexLocker(self.worker.mutex):
            self.worker.controller.set_gains(
                kp=p["kp"],
                ki=p["ki"],
                kd=p["kd"],
                dead_zone_deg=p["dead_zone_deg"],
            )
            self.worker.tracker.horizon_steps = p["prediction_steps"]
        self.plots.set_deadzone_threshold(p["dead_zone_deg"])

    def _on_manual_jog_drag(self, delta_pan: float, delta_tilt: float) -> None:
        with QMutexLocker(self.worker.mutex):
            self.worker.manual_jog_rates = (deg2rad(delta_pan), deg2rad(delta_tilt))

    def _on_manual_jog_buttons(self, rate_pan: float, rate_tilt: float) -> None:
        with QMutexLocker(self.worker.mutex):
            self.worker.manual_jog_rates = (deg2rad(rate_pan), deg2rad(rate_tilt))

    def _open_settings_dialog(self) -> None:
        was_paused = self.worker.paused
        self.worker.paused = True
        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.config = dlg.config
            self.worker.set_config(self.config)
            self.errors_history.clear()
            self.locked_frames_count = 0
            self.total_frames_count = 0
            self.plots.reset_buffers()
        self.worker.paused = was_paused

    def closeEvent(self, event) -> None:
        """Cleanly terminate worker thread upon window close."""
        self.worker.running = False
        self.worker.wait(1000)
        event.accept()


def launch_dashboard(config: Optional[Dict[str, Any]] = None) -> None:
    """Launch the Mission-Control PySide6 GUI."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dashboard = FSOCMissionControlDashboard(initial_config=config)
    dashboard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_dashboard()
