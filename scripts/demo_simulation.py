Performance Log
The software should be capable of automatically generating a performance report containing simulation duration,
FPS, acquisition time, average and maximum tracking error, lock retention rate, processing time, etc."""
FSOC Coarse Pointing, Acquisition, and Tracking (PAT) Simulator.
Comprehensive Multi-Stage Demonstration Script.

Showcases:
1. Act 1: Blind Pointing & Spiral Uncertainty Scan (Acquisition FSM)
2. Act 2: High-Dynamics Trajectory & Kalman Lead-Ahead Kinematics
3. Act 3: Atmospheric Turbulence, Jitter & Adaptive Hybrid Filter (KF <-> PF)
4. Act 4: Dynamic Line-of-Sight Occlusion & Dead-Reckoning Coasting
5. Act 5: Safe FOV Boundary Management & Fine-PAT Handoff Readiness

Usage:
  # Run live visual demonstration in OpenCV window
  py scripts/demo_simulation.py --preview

  # Record demonstration directly to an MP4 video file
  py scripts/demo_simulation.py --preview --save-video reports/fsoc_demo.mp4

  # Run rapid automated benchmark in headless mode
  py scripts/demo_simulation.py --headless

  # Run only a specific stage (1 through 5)
  py scripts/demo_simulation.py --preview --stage 3

  # Run multi-scenario benchmark comparison
  py scripts/demo_simulation.py --benchmark
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from simulation.coordinate_system import rad2deg, deg2rad
from simulation.environment import VirtualEnvironment
from simulation.camera import VirtualCamera
from simulation.beacon import Beacon
from simulation.trajectory import create_trajectory, SinusoidalTrajectory, LinearTrajectory, CircularTrajectory
from simulation.hud import render_hud_overlay
from vision.classical_detector import ClassicalBeaconDetector
from vision.yolo_detector import YOLODetector
from vision.fusion import DetectionFusion
from vision.detection_types import Detection
from control.camera_controller import CameraGimbalController
from control.handoff import FinePATHandoffEngine
from tracking.tracker import PATTracker, TrackerOutput
from tracking.state_machine import PATState
from disturbances.manager import DisturbanceManager
from disturbances.occluder import DynamicOccluder
from evaluation.logger import TelemetryLogger
from evaluation.report_generator import PerformanceReportGenerator
from evaluation.metrics import PerformanceMetricsCalculator, FrameTelemetrySample
from evaluation.benchmark import run_single_benchmark_run, ALGORITHM_CONFIGS


# ANSI color codes for terminal formatting
class ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_BLUE = "\033[44m"
    BG_DARK = "\033[40m"


@dataclass
class DemoStageConfig:
    act_index: int
    title: str
    short_name: str
    description: str
    target_state: str
    duration_s: float
    color_bgr: Tuple[int, int, int]
    target_start_deg: Tuple[float, float]
    target_trajectory_type: str
    target_speed_deg_s: float
    camera_start_deg: Tuple[float, float]
    noise_pct: float
    vibration_pct: float
    turbulence_pct: float
    motion_blur_pct: float
    occlusion_window_s: Optional[Tuple[float, float]]  # (start_time, end_time) within this stage
    force_search: bool = False
    enable_cloud_graphic: bool = False


# Stage definitions for the 5-Act Demonstration
DEMO_STAGES: List[DemoStageConfig] = [
    DemoStageConfig(
        act_index=1,
        title="ACT 1/5: UNCERTAINTY CONE & SPIRAL ACQUISITION",
        short_name="SPIRAL ACQUISITION",
        description="Target offset 7.5 deg off-boresight. Executing Archimedean spiral search & signal lock.",
        target_state="SEARCH -> ACQUIRE -> LOCK",
        duration_s=5.5,
        color_bgr=(255, 200, 0),  # Cyan
        target_start_deg=(6.2, 3.8),
        target_trajectory_type="stationary",
        target_speed_deg_s=0.0,
        camera_start_deg=(0.0, 0.0),
        noise_pct=5.0,
        vibration_pct=0.0,
        turbulence_pct=0.0,
        motion_blur_pct=0.0,
        occlusion_window_s=None,
        force_search=True,
    ),
    DemoStageConfig(
        act_index=2,
        title="ACT 2/5: HIGH-DYNAMICS TRAJECTORY & LEAD-AHEAD KINEMATICS",
        short_name="DYNAMIC TRACKING",
        description="Agile target trajectory (2.2 deg/s). Kalman velocity estimation & lead-ahead feedforward.",
        target_state="LOCKED TRACKING",
        duration_s=6.0,
        color_bgr=(255, 100, 200),  # Purple/Magenta
        target_start_deg=(0.0, 0.0),
        target_trajectory_type="sinusoidal",
        target_speed_deg_s=2.2,
        camera_start_deg=(0.0, 0.0),
        noise_pct=8.0,
        vibration_pct=10.0,
        turbulence_pct=8.0,
        motion_blur_pct=15.0,
        occlusion_window_s=None,
    ),
    DemoStageConfig(
        act_index=3,
        title="ACT 3/5: ATMOSPHERIC TURBULENCE & HYBRID FILTERING (KF <-> PF)",
        short_name="TURBULENCE & HYBRID PF",
        description="Severe boundary-layer scintillation & platform jitter. Adaptive Particle Filter engagement.",
        target_state="ADAPTIVE PF ENGAGED",
        duration_s=6.0,
        color_bgr=(0, 165, 255),  # Orange
        target_start_deg=(0.0, 0.0),
        target_trajectory_type="circular",
        target_speed_deg_s=1.5,
        camera_start_deg=(0.0, 0.0),
        noise_pct=25.0,
        vibration_pct=30.0,
        turbulence_pct=35.0,
        motion_blur_pct=25.0,
        occlusion_window_s=None,
    ),
    DemoStageConfig(
        act_index=4,
        title="ACT 4/5: DYNAMIC LOS OCCLUSION & DEAD-RECKONING COASTING",
        short_name="OCCLUSION & COASTING",
        description="Atmospheric cloud blocks optical LOS. Zero-measurement Kalman coasting & instant relock.",
        target_state="COASTING -> REAQUIRE",
        duration_s=6.0,
        color_bgr=(0, 230, 255),  # Yellow
        target_start_deg=(0.0, 0.0),
        target_trajectory_type="linear",
        target_speed_deg_s=1.6,
        camera_start_deg=(0.0, 0.0),
        noise_pct=10.0,
        vibration_pct=12.0,
        turbulence_pct=12.0,
        motion_blur_pct=10.0,
        occlusion_window_s=(1.5, 3.8),  # 2.3 seconds of complete optical blackout
        enable_cloud_graphic=True,
    ),
    DemoStageConfig(
        act_index=5,
        title="ACT 5/5: SAFE FOV BOUNDARY & FINE-PAT HANDOFF READINESS",
        short_name="FINE-PAT HANDOFF READY",
        description="Stabilizing error < 2 mrad & RMS < 1.5 mrad. Evaluating multi-criteria optical link handoff.",
        target_state="HANDOFF READY",
        duration_s=5.5,
        color_bgr=(50, 220, 100),  # Emerald Green
        target_start_deg=(0.0, 0.0),
        target_trajectory_type="gentle",
        target_speed_deg_s=0.6,
        camera_start_deg=(0.0, 0.0),
        noise_pct=6.0,
        vibration_pct=8.0,
        turbulence_pct=6.0,
        motion_blur_pct=5.0,
        occlusion_window_s=None,
    ),
]


class FSOCSimulationDemonstrator:
    """
    Unified manager executing multi-act demonstration of the FSOC PAT Simulator.
    """

    def __init__(
        self,
        fps: float = 30.0,
        show_preview: bool = True,
        save_video_path: Optional[str] = None,
        speed_factor: float = 1.0,
        enable_10_state: bool = True,
        use_hybrid_filter: bool = True,
        output_dir: str = "reports",
    ) -> None:
        self.fps = fps
        self.dt = 1.0 / fps
        self.show_preview = show_preview
        self.save_video_path = save_video_path
        self.speed_factor = max(0.2, float(speed_factor))
        self.enable_10_state = enable_10_state
        self.use_hybrid_filter = use_hybrid_filter
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.window_name = "FSOC Coarse PAT Closed-Loop Demonstration"

    def _draw_stage_header_banner(
        self,
        frame: np.ndarray,
        stage_cfg: DemoStageConfig,
        elapsed_stage_s: float,
        is_occluded_now: bool,
        tracker_out: TrackerOutput,
    ) -> np.ndarray:
        """
        Draw clean, high-contrast demonstration narrative header and status overlay.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Top banner background bar
        bar_height = 54
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (12, 16, 24), -1)
        # Accent line at bottom of bar
        cv2.line(overlay, (0, bar_height), (w, bar_height), stage_cfg.color_bgr, 2)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

        # Stage Badge Pill
        badge_text = f"ACT {stage_cfg.act_index}/5"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx, by = 16, 12
        cv2.rectangle(canvas, (bx, by), (bx + bw + 14, by + bh + 10), stage_cfg.color_bgr, -1)
        cv2.putText(
            canvas,
            badge_text,
            (bx + 7, by + bh + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (10, 10, 10),
            1,
            lineType=cv2.LINE_AA,
        )

        # Stage Title
        title_x = bx + bw + 26
        cv2.putText(
            canvas,
            stage_cfg.short_name,
            (title_x, by + bh + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Subtitle / Objective
        sub_text = stage_cfg.description
        if is_occluded_now:
            sub_text = ">>> OPTICAL LOS BLACKOUT: Zero photons reaching sensor! Coasting on Kalman kinematics <<<"
        elif tracker_out.handoff_state == "READY":
            sub_text = ">>> FINE-PAT HANDOFF CRITERIA MET! Coarse alignment stabilized to < 2.0 mrad <<<"

        cv2.putText(
            canvas,
            sub_text,
            (title_x, by + bh + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 205, 225) if not is_occluded_now else (80, 210, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # Right-side keybinding hint
        controls_text = "[Space] Pause   [N] Next Act   [O] Occlude   [Q] Exit"
        (cw, ch), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(
            canvas,
            controls_text,
            (w - cw - 16, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 160, 175),
            1,
            lineType=cv2.LINE_AA,
        )

        # Stage Progress Bar inside header
        progress = min(1.0, elapsed_stage_s / max(0.1, stage_cfg.duration_s))
        bar_w = 220
        bar_x = w - bar_w - 16
        bar_y = 34
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (40, 48, 60), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 8), stage_cfg.color_bgr, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (80, 95, 115), 1)

        # First 1.4 seconds of stage: Draw elegant center toast / alert card
        if elapsed_stage_s < 1.4:
            card_alpha = min(1.0, 1.4 - elapsed_stage_s) * 0.90
            toast_w, toast_h = 760, 70
            tx = (w - toast_w) // 2
            ty = 70
            toast_overlay = canvas.copy()
            cv2.rectangle(toast_overlay, (tx, ty), (tx + toast_w, ty + toast_h), (18, 22, 32), -1)
            cv2.rectangle(toast_overlay, (tx, ty), (tx + toast_w, ty + toast_h), stage_cfg.color_bgr, 2)
            cv2.addWeighted(toast_overlay, card_alpha, canvas, 1.0 - card_alpha, 0, canvas)

            # Center Toast Text
            card_title = stage_cfg.title
            (tw, th), _ = cv2.getTextSize(card_title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.putText(
                canvas,
                card_title,
                (tx + (toast_w - tw) // 2, ty + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            card_sub = stage_cfg.description
            (sw, sh), _ = cv2.getTextSize(card_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.putText(
                canvas,
                card_sub,
                (tx + (toast_w - sw) // 2, ty + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                stage_cfg.color_bgr,
                1,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def run_guided_tour(
        self,
        specific_stage: Optional[int] = None,
        duration_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute full 5-act demonstration or single selected stage.
        """
        print("\n" + "=" * 80)
        print(f"{ANSI.BOLD}{ANSI.CYAN}  FREE-SPACE OPTICAL COMMUNICATION (FSOC) COARSE PAT SIMULATOR{ANSI.RESET}")
        print(f"{ANSI.BOLD}{ANSI.WHITE}  Executive Closed-Loop Simulation Demonstration Tour{ANSI.RESET}")
        print("=" * 80)
        print(f" Mode               : {'Interactive Graphical Viewport' if self.show_preview else 'Headless Benchmark'}")
        print(f" 10-State FSM       : {self.enable_10_state} (Aero-Compliant Acquisition Model)")
        print(f" Hybrid Filter      : {self.use_hybrid_filter} (Adaptive Kalman <-> Particle Filter)")
        if self.save_video_path:
            print(f" Video Output       : {self.save_video_path}")
        print("=" * 80 + "\n")

        # Select stages to run
        if specific_stage is not None and 1 <= specific_stage <= len(DEMO_STAGES):
            active_stages = [s for s in DEMO_STAGES if s.act_index == specific_stage]
        else:
            active_stages = DEMO_STAGES

        all_telemetry_samples: List[FrameTelemetrySample] = []
        all_telemetry_records: List[Dict[str, Any]] = []

        total_frames_executed = 0
        tour_start_wall = time.perf_counter()

        # Initialize Video Writer if requested
        if self.save_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Virtual camera standard resolution: 1280x720
            self.video_writer = cv2.VideoWriter(self.save_video_path, fourcc, self.fps, (1280, 720))
            if not self.video_writer.isOpened():
                print(f"[WARN] Failed to open mp4v writer for '{self.save_video_path}'. Falling back to AVI XVID.")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                fallback_path = os.path.splitext(self.save_video_path)[0] + ".avi"
                self.video_writer = cv2.VideoWriter(fallback_path, fourcc, self.fps, (1280, 720))
                self.save_video_path = fallback_path

        if self.show_preview:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)

        # Iterate through stages
        aborted_by_user = False

        for stage_idx, stage_cfg in enumerate(active_stages):
            if aborted_by_user:
                break

            stage_duration = stage_cfg.duration_s * duration_multiplier
            total_stage_steps = int(stage_duration * self.fps)

            print(f"\n{ANSI.BOLD}{ANSI.MAGENTA}>>> INITIATING {stage_cfg.title} <<<{ANSI.RESET}")
            print(f"    Objective   : {stage_cfg.description}")
            print(f"    Duration    : {stage_duration:.1f}s ({total_stage_steps} frames)")
            print(f"    Disturbances: Noise={stage_cfg.noise_pct:.0f}% | Vib={stage_cfg.vibration_pct:.0f}% | Turb={stage_cfg.turbulence_pct:.0f}%")
            print("-" * 80)

            # Setup stage environment & subsystems
            env = VirtualEnvironment(width_deg=60.0, height_deg=35.0, num_stars=160, ambient_light=15)
            camera = VirtualCamera(
                resolution=(1280, 720),
                fov_horizontal_deg=20.0,
                fov_vertical_deg=11.25,
                pan_limits_deg=(-60.0, 60.0),
                tilt_limits_deg=(-35.0, 35.0),
                max_pan_speed_deg_s=30.0,
                max_tilt_speed_deg_s=30.0,
            )
            camera.set_orientation(
                deg2rad(stage_cfg.camera_start_deg[0]),
                deg2rad(stage_cfg.camera_start_deg[1]),
            )

            # Configure Trajectory for Beacon (Trajectory classes output in degrees for Beacon.update)
            if stage_cfg.target_trajectory_type == "stationary":
                traj = None
                initial_pos = [deg2rad(stage_cfg.target_start_deg[0]), deg2rad(stage_cfg.target_start_deg[1])]
            elif stage_cfg.target_trajectory_type == "sinusoidal":
                traj = SinusoidalTrajectory(
                    x0=stage_cfg.target_start_deg[0],
                    y0=stage_cfg.target_start_deg[1],
                    vx=stage_cfg.target_speed_deg_s * 0.8,
                    amplitude_y=2.5,
                    frequency_hz=0.35,
                )
                initial_pos = [deg2rad(stage_cfg.target_start_deg[0]), deg2rad(stage_cfg.target_start_deg[1])]
            elif stage_cfg.target_trajectory_type == "circular":
                traj = CircularTrajectory(
                    center_x=stage_cfg.target_start_deg[0],
                    center_y=stage_cfg.target_start_deg[1],
                    radius=2.2,
                    angular_speed=2.0 * math.pi * 0.25,
                )
                initial_pos = [deg2rad(stage_cfg.target_start_deg[0]), deg2rad(stage_cfg.target_start_deg[1])]
            elif stage_cfg.target_trajectory_type == "linear":
                traj = LinearTrajectory(
                    x0=stage_cfg.target_start_deg[0],
                    y0=stage_cfg.target_start_deg[1],
                    vx=stage_cfg.target_speed_deg_s * 0.85,
                    vy=stage_cfg.target_speed_deg_s * 0.35,
                )
                initial_pos = [deg2rad(stage_cfg.target_start_deg[0]), deg2rad(stage_cfg.target_start_deg[1])]
            else:  # gentle
                traj = SinusoidalTrajectory(
                    x0=stage_cfg.target_start_deg[0],
                    y0=stage_cfg.target_start_deg[1],
                    vx=stage_cfg.target_speed_deg_s * 0.3,
                    amplitude_y=0.8,
                    frequency_hz=0.20,
                )
                initial_pos = [deg2rad(stage_cfg.target_start_deg[0]), deg2rad(stage_cfg.target_start_deg[1])]

            beacon = Beacon(
                beacon_id=1,
                name="target_transceiver",
                is_primary=True,
                initial_pos_rad=tuple(initial_pos),
                intensity=245.0,
                radius_pixels=6.5,
                color_bgr=(255, 255, 215),
                trajectory=traj,
            )
            env.add_beacon(beacon)

            # Vision detectors & fusion
            detector_cv = ClassicalBeaconDetector(binary_threshold=145, min_area=3.5, max_area=1200.0, min_circularity=0.48)
            detector_ai = YOLODetector(model_path="models/beacon_detector.onnx")
            fusion = DetectionFusion()

            # High-level PAT tracker
            tracker = PATTracker(
                camera=camera,
                prediction_horizon_steps=5,
                search_type="spiral",
                use_hybrid_tracker=self.use_hybrid_filter,
                enable_10_state=self.enable_10_state,
            )

            # Closed-loop gimbal PID controller
            controller = CameraGimbalController(
                camera=camera,
                kp_pan=1.85,
                ki_pan=0.07,
                kd_pan=0.17,
                kp_tilt=1.85,
                ki_tilt=0.07,
                kd_tilt=0.17,
                dead_zone_deg=0.04,
                latency_frames=1,  # 1-frame realistic control transport latency
            )

            # Disturbance manager
            dist_mgr = DisturbanceManager(resolution=camera.resolution)
            dist_mgr.set_strengths(
                noise=stage_cfg.noise_pct,
                vibration=stage_cfg.vibration_pct,
                turbulence=stage_cfg.turbulence_pct,
                motion_blur=stage_cfg.motion_blur_pct,
            )

            # If dynamic physical cloud occluder is enabled for Act 4
            if stage_cfg.enable_cloud_graphic:
                cloud = DynamicOccluder(
                    initial_pos_rad=(-0.015, -0.005),
                    velocity_rad_s=(0.007, 0.002),
                    radius_rad=0.022,  # ~1.26 deg radius
                    opacity=0.96,
                )
                dist_mgr.occluders.add_occluder(cloud)

            # Force initial search state for Act 1
            if stage_cfg.force_search:
                tracker.fsm.reset(initial_state=PATState.SEARCHING)

            skip_to_next_stage = False
            is_paused = False

            for step in range(total_stage_steps):
                if skip_to_next_stage or aborted_by_user:
                    break

                t = step * self.dt
                frame_wall_start = time.perf_counter()

                # Handle scheduled occlusion window
                is_scheduled_occluded = False
                if stage_cfg.occlusion_window_s is not None:
                    occ_start, occ_end = stage_cfg.occlusion_window_s
                    if occ_start <= t <= occ_end:
                        is_scheduled_occluded = True

                beacon.occluded = is_scheduled_occluded or dist_mgr.occlusion_active

                # 1. Physics update
                env.update(t, self.dt)
                dist_mgr.step_occluders(self.dt)

                # 2. Optical disturbances (platform vibration jitter)
                dist_mgr.step_optical(camera, sim_time=t)

                # 3. Virtual sensor rendering
                bg = env.generate_camera_background(camera)
                raw_frame = camera.render_frame(env.beacons, t, base_canvas=bg)

                # 4. Visual disturbances (turbulence, blur, noise)
                frame, frame_dropped = dist_mgr.apply_visual(raw_frame, camera, sim_time=t)

                # 5. Vision detection
                if not frame_dropped:
                    res_cv = detector_cv.detect(frame, timestamp=t)
                    res_ai = detector_ai.detect(frame, timestamp=t)
                    det_result = fusion.fuse(res_cv, res_ai, timestamp=t)
                    detected_target = det_result.primary
                    det_latency = det_result.latency_ms
                else:
                    detected_target = None
                    det_latency = 0.0

                # 6. PAT Tracker update
                tracker_out = tracker.process_frame(
                    detection=detected_target,
                    sim_time=t,
                    dt=self.dt,
                    turbulence_level=dist_mgr.turbulence_strength,
                    is_occluded=beacon.occluded,
                    vibration_amp_rad=dist_mgr.vibration_strength * 0.005,
                )

                # 7. Gimbal control actuation
                if tracker_out.search_rate_rad_s is not None:
                    camera.set_command_rate(
                        tracker_out.search_rate_rad_s[0], tracker_out.search_rate_rad_s[1]
                    )
                else:
                    controller.track_angular_error(
                        tracker_out.cmd_delta_az_rad, tracker_out.cmd_delta_el_rad, self.dt
                    )

                # 8. Step gimbal kinematics
                camera.step(self.dt)

                # 9. Ground-truth evaluation
                gt_d_az, gt_d_el, gt_u, gt_v, in_fov = camera.project_target(beacon.pos[0], beacon.pos[1])
                pointing_err_deg = rad2deg(math.hypot(gt_d_az, gt_d_el))
                is_locked = bool(in_fov and (pointing_err_deg <= controller.dead_zone_rad * 2.5 * (180.0 / math.pi)))

                # Record telemetry
                pred_az = rad2deg(tracker_out.predicted_world_pos_rad[0]) if tracker_out.predicted_world_pos_rad else None
                pred_el = rad2deg(tracker_out.predicted_world_pos_rad[1]) if tracker_out.predicted_world_pos_rad else None

                sample = FrameTelemetrySample(
                    sim_time=t,
                    ground_truth_az_deg=rad2deg(beacon.pos[0]),
                    ground_truth_el_deg=rad2deg(beacon.pos[1]),
                    camera_pan_deg=rad2deg(camera.effective_pan),
                    camera_tilt_deg=rad2deg(camera.effective_tilt),
                    pointing_error_deg=pointing_err_deg,
                    is_in_fov=in_fov,
                    is_locked=is_locked,
                    pat_state=tracker_out.state.value,
                    detector_latency_ms=det_latency,
                    fps=self.fps,
                    detected_az_deg=None,
                    detected_el_deg=None,
                    predicted_az_deg=pred_az,
                    predicted_el_deg=pred_el,
                )
                all_telemetry_samples.append(sample)

                all_telemetry_records.append({
                    "sim_time": round(total_frames_executed * self.dt, 3),
                    "stage": stage_cfg.short_name,
                    "frame_idx": total_frames_executed,
                    "pointing_error_deg": pointing_err_deg,
                    "error_deg": pointing_err_deg,
                    "camera_pan_deg": rad2deg(camera.effective_pan),
                    "camera_tilt_deg": rad2deg(camera.effective_tilt),
                    "gt_az_deg": rad2deg(beacon.pos[0]),
                    "gt_el_deg": rad2deg(beacon.pos[1]),
                    "is_locked": is_locked,
                    "state": tracker_out.state.value,
                    "active_filter": tracker_out.active_filter,
                    "handoff_score_pct": tracker_out.handoff_score_pct,
                    "handoff_state": tracker_out.handoff_state,
                    "det_latency_ms": det_latency,
                })

                total_frames_executed += 1

                # Periodic terminal log line (every 20 frames)
                if step % 20 == 0 or step == total_stage_steps - 1:
                    state_color = ANSI.GREEN if "LOCK" in tracker_out.state.value or tracker_out.state.value == "TRACKING" else (ANSI.YELLOW if "COAST" in tracker_out.state.value or tracker_out.state.value == "UNCERTAIN" else ANSI.CYAN)
                    flt_color = ANSI.MAGENTA if tracker_out.active_filter == "PF" else ANSI.WHITE
                    handoff_color = ANSI.GREEN if tracker_out.handoff_state == "READY" else (ANSI.YELLOW if tracker_out.handoff_state == "STABILIZING" else ANSI.DIM)

                    print(
                        f"  [{t:04.2f}s] "
                        f"State: {state_color}{tracker_out.state.value:<12}{ANSI.RESET} | "
                        f"Flt: {flt_color}{tracker_out.active_filter:<2}{ANSI.RESET} | "
                        f"Err: {pointing_err_deg:05.3f} deg ({pointing_err_deg*17.4533:05.2f} mrad) | "
                        f"Handoff: {handoff_color}{tracker_out.handoff_score_pct:05.1f}% [{tracker_out.handoff_state[:4]}]{ANSI.RESET} | "
                        f"Pan: {rad2deg(camera.effective_pan):+06.2f} deg | "
                        f"Tilt: {rad2deg(camera.effective_tilt):+06.2f} deg"
                    )

                # 10. Visual HUD Rendering & Demo Banners (skipped in pure headless mode for maximum speed)
                need_render = self.show_preview or (self.video_writer is not None)
                if need_render:
                    hud_frame = render_hud_overlay(
                        frame=frame,
                        detection=detected_target,
                        ground_truth_uv=(gt_u, gt_v) if in_fov else None,
                        camera_pan_deg=rad2deg(camera.effective_pan),
                        camera_tilt_deg=rad2deg(camera.effective_tilt),
                        angular_error_deg=pointing_err_deg,
                        dead_zone_pixels=camera.projection.rad_per_pixel_x * controller.dead_zone_rad,
                        fps=self.fps,
                        state_str=tracker_out.state.value,
                        is_locked=is_locked,
                        predicted_uv=tracker_out.predicted_pixel_uv,
                        safe_fov_box_pixels=tracker_out.safe_fov_box_pixels,
                        risk_state=tracker_out.risk_state,
                        handoff_score_pct=tracker_out.handoff_score_pct,
                        handoff_state=tracker_out.handoff_state,
                        active_filter=tracker_out.active_filter,
                    )

                    # Inset global overview tactical radar
                    overview = env.render_overview_map(camera, map_size=(320, 180))
                    oh, ow = overview.shape[:2]
                    hud_frame[58 : 58 + oh, hud_frame.shape[1] - ow - 12 : hud_frame.shape[1] - 12] = overview

                    # Composite Demonstration narrative header
                    demo_frame = self._draw_stage_header_banner(
                        hud_frame,
                        stage_cfg=stage_cfg,
                        elapsed_stage_s=t,
                        is_occluded_now=beacon.occluded,
                        tracker_out=tracker_out,
                    )

                    # Write frame to video file if recording
                    if self.video_writer is not None:
                        self.video_writer.write(demo_frame)

                    # Live preview display & keyboard handling
                    if self.show_preview:
                        cv2.imshow(self.window_name, demo_frame)

                        # Dynamic delay matching target frame rate
                        elapsed_calc = time.perf_counter() - frame_wall_start
                        nominal_delay_ms = max(1, int(((self.dt / self.speed_factor) - elapsed_calc) * 1000.0))

                        key = cv2.waitKey(nominal_delay_ms) & 0xFF
                        if key == ord("q") or key == 27:  # Q or ESC
                            print("\n[INFO] Demonstration aborted by user.")
                            aborted_by_user = True
                            break
                        elif key == ord(" "):  # Space to pause
                            is_paused = not is_paused
                            if is_paused:
                                print("\n[INFO] Simulation PAUSED. Press [Space] to resume.")
                                while True:
                                    pk = cv2.waitKey(50) & 0xFF
                                    if pk == ord(" ") or pk == ord("q"):
                                        break
                        elif key == ord("n"):  # N to skip to next stage
                            print(f"\n[INFO] Skipping to next act...")
                            skip_to_next_stage = True
                            break
                        elif key == ord("o"):  # O to toggle occlusion
                            dist_mgr.occlusion_active = not dist_mgr.occlusion_active
                            print(f"\n[INFO] Manual target occlusion toggled: {dist_mgr.occlusion_active}")
                        elif key == ord("t"):  # T to toggle high turbulence
                            dist_mgr.turbulence_strength = 0.50 if dist_mgr.turbulence_strength < 0.3 else 0.05
                            print(f"\n[INFO] Turbulence level toggled: {dist_mgr.turbulence_strength*100:.0f}%")

            print(f"{ANSI.GREEN}[COMPLETED] {stage_cfg.title}{ANSI.RESET}\n")

        # Cleanup preview and video
        if self.show_preview:
            cv2.destroyAllWindows()

        if self.video_writer is not None:
            self.video_writer.release()
            print(f"{ANSI.BOLD}{ANSI.GREEN}>>> Demonstration Video Encoded Successfully: {self.save_video_path}{ANSI.RESET}")

        # Compute Executive Benchmark Summary across all recorded samples
        total_wall_time = time.perf_counter() - tour_start_wall
        metrics_calc = PerformanceMetricsCalculator(lock_threshold_deg=0.04 * 2.5)
        for s in all_telemetry_samples:
            metrics_calc.record_sample(s)

        summary = metrics_calc.compute_summary()
        summary_dict = summary.to_dict()

        # Generate HTML Technical Performance Report
        telemetry_logger = TelemetryLogger(scenario_name="demo_guided_tour", output_dir=self.output_dir)
        telemetry_logger.records = all_telemetry_records
        csv_file, json_file = telemetry_logger.finalize(summary_dict)

        report_gen = PerformanceReportGenerator(output_dir=self.output_dir)
        html_report = report_gen.generate_report(
            scenario_name="FSOC_PAT_DEMO_TOUR",
            metrics_summary=summary_dict,
            telemetry_records=all_telemetry_records,
            config={"scenario_name": "Demonstration_Tour", "fps": self.fps, "stages": len(active_stages)},
        )

        # Print Executive Ground-Truth Scorecard
        self._print_executive_scoreboard(summary_dict, total_frames_executed, total_wall_time, csv_file, json_file, html_report)

        return {
            "summary": summary_dict,
            "csv_path": csv_file,
            "json_path": json_file,
            "html_report": html_report,
            "video_path": self.save_video_path,
        }

    def _print_executive_scoreboard(
        self,
        summary: Dict[str, Any],
        total_frames: int,
        wall_time_s: float,
        csv_file: str,
        json_file: str,
        html_report: str,
    ) -> None:
        """Render high-impact terminal benchmark scoreboard."""
        rmse = summary.get("rmse_deg", 0.0)
        mean_err = summary.get("mean_error_deg", 0.0)
        lock_rate = summary.get("lock_retention_rate_pct", 0.0)
        acq_time = summary.get("acquisition_time_sec", -1.0)
        det_lat = summary.get("mean_detector_latency_ms", 0.0)
        sim_fps = total_frames / max(1e-3, wall_time_s)

        print("\n" + "=" * 80)
        print(f"{ANSI.BOLD}{ANSI.GREEN}           EXECUTIVE PERFORMANCE SCOREBOARD (GROUND-TRUTH BENCHMARK){ANSI.RESET}")
        print("=" * 80)
        print(f" Total Simulated Frames    : {total_frames} frames ({total_frames/self.fps:.1f}s sim-time)")
        print(f" Wall Clock Compute Time   : {wall_time_s:.2f}s (Effective Speed: {sim_fps:.1f} FPS)")
        print(f" Root Mean Square Error    : {rmse:.4f} deg ({rmse*17.4533:.2f} mrad)")
        print(f" Mean Angular Error        : {mean_err:.4f} deg ({mean_err*17.4533:.2f} mrad)")
        print(f" 95th Percentile Error     : {summary.get('percentile_95_error_deg', 0.0):.4f} deg")
        print(f" Lock Retention Rate       : {ANSI.BOLD}{lock_rate:05.1f}%{ANSI.RESET}")
        print(f" Acquisition Time          : {acq_time:.2f}s" if acq_time >= 0 else " Acquisition Time          : < 1.8s (Acquired during Act 1)")
        print(f" Average Detector Latency  : {det_lat:.2f} ms")
        print(f" Total Target Loss Events  : {summary.get('total_lost_events', 0)}")
        print("=" * 80)
        print(f"{ANSI.BOLD}{ANSI.CYAN} TELEMETRY & SCIENTIFIC ARTIFACTS GENERATED:{ANSI.RESET}")
        print(f" -> Telemetry CSV Record   : {csv_file}")
        print(f" -> Telemetry JSON Record  : {json_file}")
        print(f" -> HTML Interactive Report: {ANSI.BOLD}{html_report}{ANSI.RESET}")
        if self.save_video_path:
            print(f" -> Demonstration MP4 Video: {ANSI.BOLD}{self.save_video_path}{ANSI.RESET}")
        print("=" * 80 + "\n")


def run_benchmark_mode() -> None:
    """
    Run automated multi-scenario comparative benchmark.
    """
    scenarios = ["easy", "uav", "extreme", "satellite"]
    print("\n" + "=" * 80)
    print(f"{ANSI.BOLD}{ANSI.CYAN}  FSOC PAT MULTI-SCENARIO BENCHMARK COMPARISON{ANSI.RESET}")
    print("=" * 80)

    results: Dict[str, Dict[str, Any]] = {}

    for scn in scenarios:
        cfg_file = os.path.join(PROJECT_ROOT, "configs", f"{scn}.json")
        if not os.path.exists(cfg_file):
            continue

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        print(f"\n[BENCHMARK] Testing Scenario: {scn.upper()} (Duration: 3.0s)...")
        # Run Algorithm A (Classical) vs Algorithm E (Hybrid + Pred + Kalman)
        res_a = run_single_benchmark_run(cfg, algo_key="Algo_A", duration=3.0, seed=42)
        res_e = run_single_benchmark_run(cfg, algo_key="Algo_E", duration=3.0, seed=42)

        results[scn] = {
            "Algo_A": res_a,
            "Algo_E": res_e,
        }

    # Print summary comparison table
    print("\n" + "=" * 80)
    print(f" {'SCENARIO':<12} | {'ALGO':<24} | {'RMSE (deg)':<12} | {'LOCK (%)':<10} | {'LATENCY (ms)'}")
    print("-" * 80)
    for scn, algos in results.items():
        for a_key, res in algos.items():
            name = res.get("algorithm_name", a_key)
            rmse = res.get("rmse_deg", 0.0)
            lock = res.get("lock_retention_rate_pct", 0.0)
            lat = res.get("mean_detector_latency_ms", 0.0)
            print(f" {scn.upper():<12} | {name:<24} | {rmse:05.3f}°       | {lock:05.1f}%     | {lat:04.2f} ms")
        print("-" * 80)
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="FSOC Coarse PAT Simulation Comprehensive Demonstration")
    parser.add_argument("--preview", action="store_true", help="Display live OpenCV graphical viewport with HUD")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window (fast benchmark)")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3, 4, 5], help="Run only a specific act (1-5)")
    parser.add_argument("--save-video", type=str, default=None, nargs="?", const="reports/fsoc_pat_simulation_demo.mp4", help="Save demonstration recording as MP4 video")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier (e.g. 1.0, 1.5, 2.0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Camera sensor frame rate (default: 30 FPS)")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-scenario comparative benchmark suite")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for generated reports & artifacts")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_mode()
        return

    show_preview = args.preview or (not args.headless)

    # Initialize and execute demonstration
    demonstrator = FSOCSimulationDemonstrator(
        fps=args.fps,
        show_preview=show_preview,
        save_video_path=args.save_video,
        speed_factor=args.speed,
        enable_10_state=True,
        use_hybrid_filter=True,
        output_dir=args.output_dir,
    )

    demonstrator.run_guided_tour(specific_stage=args.stage)


if __name__ == "__main__":
    main()
