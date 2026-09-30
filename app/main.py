"""
FSOC Coarse Pointing, Acquisition, and Tracking (PAT) Simulator.
Main CLI & Demonstration Entry Point.

Usage:
  py app/main.py --headless --config configs/default.json --duration 10.0
  py app/main.py --headless --config configs/uav.json --duration 10.0
  py app/main.py --preview --config configs/easy.json --duration 15.0
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

# Ensure root package is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.coordinate_system import rad2deg
from simulation.environment import build_environment_from_config
from simulation.hud import render_hud_overlay
from vision.classical_detector import ClassicalBeaconDetector
from vision.yolo_detector import YOLODetector
from vision.fusion import DetectionFusion
from control.camera_controller import CameraGimbalController
from tracking.tracker import PATTracker
from tracking.state_machine import PATState
from disturbances.manager import DisturbanceManager
from evaluation.logger import TelemetryLogger
from evaluation.report_generator import PerformanceReportGenerator
from evaluation.metrics import PerformanceMetricsCalculator, FrameTelemetrySample


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration with fallback."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)

    if not os.path.exists(config_path):
        print(f"[WARN] Config file '{config_path}' not found. Falling back to configs/default.json")
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.json")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_simulation(
    config: Dict[str, Any],
    duration: float = 10.0,
    show_preview: bool = False,
    headless: bool = True,
) -> Dict[str, float]:
    """
    Execute closed-loop coarse PAT simulation with Kalman tracking,
    predictive control, 6-state PAT state machine, and disturbance engine.
    """
    env, camera = build_environment_from_config(config)

    det_cfg = config.get("detection", {})
    detector_mode = config.get("detector_mode", "hybrid").lower()

    detector_cv = ClassicalBeaconDetector(
        binary_threshold=det_cfg.get("binary_threshold", 150),
        min_area=det_cfg.get("min_area", 4.0),
        max_area=det_cfg.get("max_area", 1200.0),
        min_circularity=det_cfg.get("min_circularity", 0.50),
    )
    detector_ai = YOLODetector(model_path=config.get("ai_model_path", "models/beacon_detector.onnx"))
    fusion = DetectionFusion()

    track_cfg = config.get("tracking", {})
    tracker = PATTracker(
        camera=camera,
        prediction_horizon_steps=track_cfg.get("prediction_horizon_steps", 5),
        search_type=track_cfg.get("search_strategy", "spiral"),
    )

    pid_cfg = config.get("pid", {})
    controller = CameraGimbalController(
        camera=camera,
        kp_pan=pid_cfg.get("kp_pan", 2.0),
        ki_pan=pid_cfg.get("ki_pan", 0.08),
        kd_pan=pid_cfg.get("kd_pan", 0.18),
        kp_tilt=pid_cfg.get("kp_tilt", 2.0),
        ki_tilt=pid_cfg.get("ki_tilt", 0.08),
        kd_tilt=pid_cfg.get("kd_tilt", 0.18),
        dead_zone_deg=pid_cfg.get("dead_zone_deg", 0.04),
    )

    dist_cfg = config.get("disturbances", {})
    dist_mgr = DisturbanceManager(resolution=camera.resolution)
    dist_mgr.set_strengths(
        noise=dist_cfg.get("sensor_noise_percent", 0.0),
        vibration=dist_cfg.get("vibration_percent", 0.0),
        turbulence=dist_cfg.get("turbulence_percent", 0.0),
        motion_blur=dist_cfg.get("motion_blur_percent", 0.0),
        frame_drop=dist_cfg.get("frame_drop_percent", 0.0),
    )

    primary_beacon = env.get_primary_beacon()
    if primary_beacon is None:
        raise RuntimeError("No optical beacon defined in environment!")

    fps = config.get("camera", {}).get("fps", 30.0)
    dt = 1.0 / fps
    total_steps = int(duration * fps)

    scenario_name = config.get("scenario_name", "DEFAULT")
    print("=" * 76)
    print(f" FREE SPACE OPTICAL COMMUNICATION (FSOC) COARSE PAT SIMULATOR [{scenario_name}]")
    print(f" Target Duration: {duration:.1f}s | FPS: {fps:.0f} | Steps: {total_steps}")
    print(f" Primary Target : '{primary_beacon.name}' (Trajectory: {config.get('beacon',{}).get('trajectory',{}).get('type')})")
    print(f" Disturbances   : Noise={dist_mgr.noise_strength*100:.0f}% | Vib={dist_mgr.vibration_strength*100:.0f}% | Turb={dist_mgr.turbulence_strength*100:.0f}%")
    print("=" * 76)

    errors_deg: List[float] = []
    locked_frames = 0
    acquisition_time_s: float = -1.0
    latencies_ms: List[float] = []
    state_history: List[str] = []

    lock_deadzone_deg = pid_cfg.get("dead_zone_deg", 0.04)

    telemetry_logger = TelemetryLogger(scenario_name=scenario_name.lower())
    metrics_calculator = PerformanceMetricsCalculator(lock_threshold_deg=lock_deadzone_deg * 2.5)

    if show_preview:
        window_title = f"FSOC PAT Simulator - [{scenario_name}]"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, 720)

    start_wall_time = time.perf_counter()

    for step in range(total_steps):
        t = step * dt
        step_start = time.perf_counter()

        # 1. Advance Environment Physics
        env.update(t, dt)

        # 2. Inject Optical Disturbances (Platform vibration & jitter)
        dist_mgr.step_optical(camera, sim_time=t)

        # 3. Render Virtual Camera Sensor Frame
        bg = env.generate_camera_background(camera)
        raw_frame = camera.render_frame(env.beacons, t, base_canvas=bg)

        # 4. Apply Visual Disturbances (Turbulence, Motion Blur, Noise, Frame Drops)
        frame, frame_dropped = dist_mgr.apply_visual(raw_frame, camera, sim_time=t)

        # 5. Vision Detection Pipeline
        if not frame_dropped:
            if detector_mode == "classical":
                det_result = detector_cv.detect(frame, timestamp=t)
            elif detector_mode == "ai":
                det_result = detector_ai.detect(frame, timestamp=t)
            else:
                # Hybrid fusion mode
                res_cv = detector_cv.detect(frame, timestamp=t)
                res_ai = detector_ai.detect(frame, timestamp=t)
                det_result = fusion.fuse(res_cv, res_ai, timestamp=t)

            latencies_ms.append(det_result.latency_ms)
            detected_target = det_result.primary
        else:
            detected_target = None
            latencies_ms.append(0.0)

        # 6. PAT Tracker (Kalman state estimation, lead-ahead prediction, 6-state FSM)
        tracker_out = tracker.process_frame(detected_target, sim_time=t, dt=dt)
        state_history.append(tracker_out.state.value)

        # 7. Closed-Loop Controller
        if tracker_out.search_rate_rad_s is not None:
            camera.set_command_rate(
                tracker_out.search_rate_rad_s[0], tracker_out.search_rate_rad_s[1]
            )
        else:
            controller.track_angular_error(
                tracker_out.cmd_delta_az_rad,
                tracker_out.cmd_delta_el_rad,
                dt,
                feedforward_vel_rad_s=tracker_out.estimated_vel_rad_s,
            )

        # 8. Step Camera Gimbal Kinematics
        camera.step(dt)

        # 9. Ground-Truth Telemetry
        gt_delta_az, gt_delta_el, gt_u, gt_v, in_fov = camera.project_target(
            primary_beacon.pos[0], primary_beacon.pos[1]
        )
        ang_error_deg = rad2deg(math.hypot(gt_delta_az, gt_delta_el))
        errors_deg.append(ang_error_deg)

        is_locked = bool(in_fov and (ang_error_deg <= lock_deadzone_deg * 2.5))
        if is_locked:
            locked_frames += 1
            if acquisition_time_s < 0:
                acquisition_time_s = t

        state_str = tracker_out.state.value

        step_fps = 1.0 / max(1e-5, time.perf_counter() - step_start)

        # Log frame telemetry & ground-truth sample
        pred_az = rad2deg(tracker_out.predicted_world_pos_rad[0]) if tracker_out.predicted_world_pos_rad is not None else None
        pred_el = rad2deg(tracker_out.predicted_world_pos_rad[1]) if tracker_out.predicted_world_pos_rad is not None else None
        det_lat = det_result.latency_ms if not frame_dropped else 0.0

        telemetry_logger.log_frame({
            "sim_time": t,
            "frame_idx": step,
            "gt_az_deg": rad2deg(primary_beacon.pos[0]),
            "gt_el_deg": rad2deg(primary_beacon.pos[1]),
            "pan_deg": rad2deg(camera.effective_pan),
            "tilt_deg": rad2deg(camera.effective_tilt),
            "error_deg": ang_error_deg,
            "is_locked": is_locked,
            "state": state_str,
            "fps": step_fps,
            "det_latency_ms": det_lat,
            "detected_u": detected_target.center_u if detected_target else None,
            "detected_v": detected_target.center_v if detected_target else None,
            "pred_az_deg": pred_az,
            "pred_el_deg": pred_el,
            "confidence": detected_target.confidence if detected_target else 0.0,
        })

        metrics_calculator.record_sample(
            FrameTelemetrySample(
                sim_time=t,
                ground_truth_az_deg=rad2deg(primary_beacon.pos[0]),
                ground_truth_el_deg=rad2deg(primary_beacon.pos[1]),
                camera_pan_deg=rad2deg(camera.effective_pan),
                camera_tilt_deg=rad2deg(camera.effective_tilt),
                pointing_error_deg=ang_error_deg,
                is_in_fov=in_fov,
                is_locked=is_locked,
                pat_state=state_str,
                detector_latency_ms=det_lat,
                fps=step_fps,
                detected_az_deg=None,
                detected_el_deg=None,
                predicted_az_deg=pred_az,
                predicted_el_deg=pred_el,
            )
        )

        # Telemetry log every 15 frames
        if step % 15 == 0 or step == total_steps - 1:
            est_vel = tracker_out.estimated_vel_rad_s
            print(
                f"[{t:05.2f}s | #{step:04d}] "
                f"State: {state_str:<10} | "
                f"Risk: {tracker_out.risk_state:<9} | "
                f"Handoff: {tracker_out.handoff_score_pct:04.1f}% [{tracker_out.handoff_state[:4]}] | "
                f"Flt: {tracker_out.active_filter} | "
                f"Err: {ang_error_deg:05.3f} deg | "
                f"Pan: {rad2deg(camera.effective_pan):+06.2f} deg | "
                f"Tilt: {rad2deg(camera.effective_tilt):+06.2f} deg | "
                f"Det: {det_result.latency_ms:04.2f}ms"
            )

        # 10. Optional Visual Preview with HUD Overlay
        if show_preview:
            hud_frame = render_hud_overlay(
                frame=frame,
                detection=detected_target,
                ground_truth_uv=(gt_u, gt_v) if in_fov else None,
                camera_pan_deg=rad2deg(camera.effective_pan),
                camera_tilt_deg=rad2deg(camera.effective_tilt),
                angular_error_deg=ang_error_deg,
                dead_zone_pixels=camera.projection.rad_per_pixel_x * controller.dead_zone_rad,
                fps=step_fps,
                state_str=state_str,
                is_locked=is_locked,
                predicted_uv=tracker_out.predicted_pixel_uv,
                safe_fov_box_pixels=tracker_out.safe_fov_box_pixels,
                risk_state=tracker_out.risk_state,
                handoff_score_pct=tracker_out.handoff_score_pct,
                handoff_state=tracker_out.handoff_state,
                active_filter=tracker_out.active_filter,
            )

            # Inset global overview tactical map
            overview = env.render_overview_map(camera, map_size=(320, 180))
            oh, ow = overview.shape[:2]
            hud_frame[48 : 48 + oh, hud_frame.shape[1] - ow - 10 : hud_frame.shape[1] - 10] = overview

            cv2.imshow(window_title, hud_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] Simulation aborted by user (Q key).")
                break
            elif key == ord(" "):
                cv2.waitKey(0)  # Pause
            elif key == ord("o"):
                # Press 'o' to toggle artificial occlusion
                primary_beacon.occluded = not primary_beacon.occluded
                print(f"\n[INFO] Target occlusion toggled: {primary_beacon.occluded}")

    if show_preview:
        cv2.destroyAllWindows()

    total_wall_time = time.perf_counter() - start_wall_time
    mean_error = float(np.mean(errors_deg)) if errors_deg else 0.0
    max_error = float(np.max(errors_deg)) if errors_deg else 0.0
    rmse = float(np.sqrt(np.mean(np.square(errors_deg)))) if errors_deg else 0.0
    lock_retention = (locked_frames / max(1, len(errors_deg))) * 100.0
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    print("\n" + "=" * 76)
    print(" EXECUTIVE PERFORMANCE EVALUATION (GROUND TRUTH BENCHMARK) ")
    print("=" * 76)
    print(f" Scenario Name             : {scenario_name}")
    print(f" Total Simulated Frames    : {len(errors_deg)}")
    print(f" Effective Simulation Rate : {len(errors_deg)/total_wall_time:.1f} FPS (Wall time: {total_wall_time:.2f}s)")
    print(f" Initial Pointing Error    : {errors_deg[0]:.3f} deg")
    print(f" Final Pointing Error      : {errors_deg[-1]:.3f} deg")
    print(f" Mean Angular Error        : {mean_error:.3f} deg")
    print(f" Peak Maximum Error        : {max_error:.3f} deg")
    print(f" Root Mean Square Error    : {rmse:.3f} deg")
    print(f" Acquisition Time          : {acquisition_time_s:.2f}s" if acquisition_time_s >= 0 else " Acquisition Time          : NOT ACQUIRED")
    print(f" Lock Retention Rate       : {lock_retention:.1f}%")
    print(f" Average Detector Latency  : {avg_latency:.2f} ms")
    print(f" Total Loss Events         : {tracker.fsm.total_lost_events}")
    print(f" Reacquisition Latency     : {tracker.fsm.last_reacquisition_latency:.2f}s" if tracker.fsm.total_reacquisitions > 0 else " Reacquisition Latency     : N/A")
    print("=" * 76)

    summary = metrics_calculator.compute_summary()
    csv_file, json_file = telemetry_logger.finalize(summary.to_dict())

    # Generate standalone interactive HTML performance report
    report_gen = PerformanceReportGenerator()
    html_report = report_gen.generate_report(
        scenario_name=scenario_name,
        metrics_summary=summary.to_dict(),
        telemetry_records=telemetry_logger.records,
        config=config,
    )

    print("\n" + "=" * 76)
    print(" TELEMETRY & REPORT ARTIFACTS GENERATED")
    print("=" * 76)
    print(f" -> Telemetry Stream CSV : {csv_file}")
    print(f" -> Telemetry Stream JSON: {json_file}")
    print(f" -> HTML Technical Report: {html_report}")
    print("=" * 76)

    res = summary.to_dict()
    res["csv_path"] = csv_file
    res["json_path"] = json_file
    res["html_report_path"] = html_report
    return res


def main():
    parser = argparse.ArgumentParser(description="FSOC Coarse PAT Virtual Camera Tracking Simulator")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to scenario config JSON")
    parser.add_argument("--scenario", type=str, default="", help="Quick scenario preset: easy, uav, satellite, extreme")
    parser.add_argument("--detector", type=str, default="hybrid", choices=["classical", "ai", "hybrid"], help="Detection engine")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Run in headless benchmark/CLI mode")
    parser.add_argument("--preview", action="store_true", help="Display OpenCV live camera HUD viewport")
    parser.add_argument("--gui", action="store_true", help="Launch full PySide6 Mission-Control GUI")

    args = parser.parse_args()

    cfg_path = args.config
    if args.scenario:
        cfg_path = f"configs/{args.scenario.lower()}.json"

    config = load_config(cfg_path)
    if args.detector:
        config["detector_mode"] = args.detector

    # If --headless or --preview is explicitly requested, run CLI simulation
    if args.headless or args.preview:
        run_simulation(
            config=config,
            duration=args.duration,
            show_preview=args.preview,
            headless=args.headless,
        )
    else:
        # Default mode: Launch the full PySide6 Mission-Control GUI Dashboard
        from app.gui.dashboard import launch_dashboard
        launch_dashboard(config)


if __name__ == "__main__":
    main()
