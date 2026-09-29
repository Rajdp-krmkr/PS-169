"""
Automated Multi-Algorithm Benchmark Suite for FSOC Coarse PAT.

Compares 5 foundational PAT tracking architectures:
- Algorithm A: Classical CV + PID
- Algorithm B: AI YOLO + PID
- Algorithm C: AI YOLO + Kalman Filter + PID
- Algorithm D: Multi-Modal Hybrid Detection + Kalman Filter + PID
- Algorithm E: Multi-Modal Hybrid Detection + Kalman Filter + Predictive Lead-Ahead + Adaptive Search
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure root package is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.coordinate_system import rad2deg
from simulation.environment import build_environment_from_config
from vision.classical_detector import ClassicalBeaconDetector
from vision.yolo_detector import YOLODetector
from vision.fusion import DetectionFusion
from control.camera_controller import CameraGimbalController
from tracking.tracker import PATTracker
from tracking.state_machine import PATState
from disturbances.manager import DisturbanceManager
from evaluation.metrics import PerformanceMetricsCalculator, FrameTelemetrySample


ALGORITHM_CONFIGS = {
    "Algo_A": {
        "name": "Algorithm A (Classical CV + PID)",
        "detector": "classical",
        "use_kalman": False,
        "use_prediction": False,
        "search_strategy": "raster",
    },
    "Algo_B": {
        "name": "Algorithm B (AI YOLO + PID)",
        "detector": "ai",
        "use_kalman": False,
        "use_prediction": False,
        "search_strategy": "raster",
    },
    "Algo_C": {
        "name": "Algorithm C (AI YOLO + Kalman + PID)",
        "detector": "ai",
        "use_kalman": True,
        "use_prediction": False,
        "search_strategy": "spiral",
    },
    "Algo_D": {
        "name": "Algorithm D (Hybrid + Kalman + PID)",
        "detector": "hybrid",
        "use_kalman": True,
        "use_prediction": False,
        "search_strategy": "spiral",
    },
    "Algo_E": {
        "name": "Algorithm E (Hybrid + Kalman + Predictive + Adaptive Search)",
        "detector": "hybrid",
        "use_kalman": True,
        "use_prediction": True,
        "search_strategy": "spiral",
    },
}


def run_single_benchmark_run(
    base_config: Dict[str, Any],
    algo_key: str,
    duration: float = 4.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single benchmark run for a specific algorithm and random seed."""
    algo_meta = ALGORITHM_CONFIGS[algo_key]
    cfg = copy.deepcopy(base_config)

    env, camera = build_environment_from_config(cfg)
    primary_beacon = env.get_primary_beacon()

    fps = cfg.get("camera", {}).get("fps", 30.0)
    dt = 1.0 / fps
    total_steps = int(duration * fps)

    det_cfg = cfg.get("detection", {})
    detector_cv = ClassicalBeaconDetector(
        binary_threshold=det_cfg.get("binary_threshold", 140),
        min_area=det_cfg.get("min_area", 3.0),
        max_area=det_cfg.get("max_area", 1200.0),
        min_circularity=det_cfg.get("min_circularity", 0.45),
    )
    detector_ai = YOLODetector(model_path=cfg.get("ai_model_path", "models/beacon_detector.onnx"))
    fusion = DetectionFusion()

    pred_steps = 6 if algo_meta["use_prediction"] else 0
    tracker = PATTracker(
        camera=camera,
        prediction_horizon_steps=pred_steps,
        search_type=algo_meta["search_strategy"],
    )

    pid_cfg = cfg.get("pid", {})
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

    dist_cfg = cfg.get("disturbances", {})
    dist_mgr = DisturbanceManager(resolution=camera.resolution, seed=seed)
    dist_mgr.set_strengths(
        noise=dist_cfg.get("sensor_noise_percent", 0.0),
        vibration=dist_cfg.get("vibration_percent", 0.0),
        turbulence=dist_cfg.get("turbulence_percent", 0.0),
        motion_blur=dist_cfg.get("motion_blur_percent", 0.0),
    )

    calculator = PerformanceMetricsCalculator(lock_threshold_deg=0.08)

    for step in range(total_steps):
        t = step * dt
        t_start = time.perf_counter()

        env.update(t, dt)
        dist_mgr.step_optical(camera, t)

        bg = env.generate_camera_background(camera)
        raw_frame = camera.render_frame(env.beacons, t, base_canvas=bg)
        frame, dropped = dist_mgr.apply_visual(raw_frame, camera, sim_time=t)

        # Vision detection
        if not dropped:
            if algo_meta["detector"] == "classical":
                det_res = detector_cv.detect(frame, timestamp=t)
            elif algo_meta["detector"] == "ai":
                det_res = detector_ai.detect(frame, timestamp=t)
            else:
                c_res = detector_cv.detect(frame, timestamp=t)
                a_res = detector_ai.detect(frame, timestamp=t)
                det_res = fusion.fuse(c_res, a_res, timestamp=t)
            detected = det_res.primary
            det_lat = det_res.latency_ms
        else:
            detected = None
            det_lat = 0.0

        # Tracking state machine
        tracker_out = tracker.process_frame(detected, sim_time=t, dt=dt)

        # Controller execution
        if not algo_meta["use_kalman"] and detected is not None:
            # Direct PID without Kalman
            controller.track_pixel_error(detected.center_u, detected.center_v, dt)
        elif tracker_out.search_rate_rad_s is not None:
            camera.set_command_rate(
                tracker_out.search_rate_rad_s[0], tracker_out.search_rate_rad_s[1]
            )
        else:
            controller.track_angular_error(
                tracker_out.cmd_delta_az_rad, tracker_out.cmd_delta_el_rad, dt
            )

        camera.step(dt)

        # Ground truth error
        gt_az, gt_el, _, _, in_fov = camera.project_target(primary_beacon.pos[0], primary_beacon.pos[1])
        pointing_err_deg = rad2deg(math.hypot(gt_az, gt_el))
        is_locked = bool(in_fov and (pointing_err_deg <= 0.08))

        step_fps = 1.0 / max(1e-5, time.perf_counter() - t_start)

        calculator.record_sample(
            FrameTelemetrySample(
                sim_time=t,
                ground_truth_az_deg=rad2deg(primary_beacon.pos[0]),
                ground_truth_el_deg=rad2deg(primary_beacon.pos[1]),
                camera_pan_deg=rad2deg(camera.effective_pan),
                camera_tilt_deg=rad2deg(camera.effective_tilt),
                pointing_error_deg=pointing_err_deg,
                is_in_fov=in_fov,
                is_locked=is_locked,
                pat_state=tracker_out.state.value,
                detector_latency_ms=det_lat,
                fps=step_fps,
            )
        )

    summary = calculator.compute_summary()
    return summary.to_dict()


def run_benchmark_suite(
    scenarios: List[str] = ["uav", "satellite", "extreme"],
    algorithms: List[str] = ["Algo_A", "Algo_C", "Algo_D", "Algo_E"],
    duration: float = 3.5,
    num_seeds: int = 3,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Run full Monte Carlo benchmark comparison matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_matrix: Dict[str, Dict[str, Any]] = {}

    print("=" * 80)
    print(" FSOC COARSE PAT AUTOMATED ALGORITHM BENCHMARK SUITE ")
    print(f" Scenarios : {scenarios}")
    print(f" Algorithms: {algorithms}")
    print(f" Runs/Seeds: {num_seeds} | Duration per run: {duration:.1f}s")
    print("=" * 80)

    for scn in scenarios:
        cfg_path = os.path.join(PROJECT_ROOT, "configs", f"{scn}.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)

        results_matrix[scn] = {}

        for algo in algorithms:
            algo_name = ALGORITHM_CONFIGS[algo]["name"]
            print(f"--> Testing [{scn.upper()}] with [{algo_name}] across {num_seeds} seeds...")

            runs_data = []
            for s_idx in range(num_seeds):
                seed = 100 + s_idx * 17
                run_res = run_single_benchmark_run(
                    base_config=base_cfg,
                    algo_key=algo,
                    duration=duration,
                    seed=seed,
                )
                runs_data.append(run_res)

            # Aggregate stats across seeds
            rmses = [r["rmse_deg"] for r in runs_data]
            mean_errs = [r["mean_error_deg"] for r in runs_data]
            lock_rates = [r["lock_retention_rate_pct"] for r in runs_data]
            latencies = [r["mean_detector_latency_ms"] for r in runs_data]
            losses = [r["total_lost_events"] for r in runs_data]

            results_matrix[scn][algo] = {
                "algorithm_name": algo_name,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "mean_error": float(np.mean(mean_errs)),
                "lock_retention_mean": float(np.mean(lock_rates)),
                "detector_latency_ms": float(np.mean(latencies)),
                "total_loss_events": int(np.sum(losses)),
                "raw_runs": runs_data,
            }

    # Print Scorecard Table
    print("\n" + "=" * 90)
    print(f"{'SCENARIO':<12} | {'ALGORITHM':<35} | {'RMSE (deg)':<12} | {'LOCK RATE':<10} | {'LATENCY':<8}")
    print("-" * 90)
    for scn, algos in results_matrix.items():
        for algo_key, m in algos.items():
            algo_short = m["algorithm_name"][:34]
            print(
                f"{scn.upper():<12} | {algo_short:<35} | "
                f"{m['rmse_mean']:05.3f}±{m['rmse_std']:04.3f} | "
                f"{m['lock_retention_mean']:5.1f}%    | "
                f"{m['detector_latency_ms']:5.1f}ms"
            )
        print("-" * 90)

    # Save to JSON and CSV
    json_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_matrix, f, indent=2)

    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Algorithm_Key", "Algorithm_Name", "RMSE_deg_Mean", "RMSE_deg_Std", "Mean_Error_deg", "Lock_Retention_Pct", "Detector_Latency_ms", "Loss_Events"])
        for scn, algos in results_matrix.items():
            for algo_key, m in algos.items():
                writer.writerow([
                    scn, algo_key, m["algorithm_name"],
                    round(m["rmse_mean"], 4), round(m["rmse_std"], 4),
                    round(m["mean_error"], 4), round(m["lock_retention_mean"], 2),
                    round(m["detector_latency_ms"], 2), m["total_loss_events"]
                ])

    print(f"\n[SUCCESS] Benchmark results saved to: {json_path} and {csv_path}")
    return results_matrix


def main():
    parser = argparse.ArgumentParser(description="Run FSOC PAT Benchmark Suite")
    parser.add_argument("--scenarios", nargs="+", default=["uav", "satellite", "extreme"], help="Scenarios to benchmark")
    parser.add_argument("--algorithms", nargs="+", default=["Algo_A", "Algo_C", "Algo_D", "Algo_E"], help="Algorithms to benchmark")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration per benchmark run")
    parser.add_argument("--seeds", type=int, default=3, help="Monte Carlo seeds per test")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (1 seed, 2.0s duration)")

    args = parser.parse_args()

    dur = 2.0 if args.quick else args.duration
    seeds = 1 if args.quick else args.seeds
    scenarios = ["uav"] if args.quick else args.scenarios
    algorithms = ["Algo_A", "Algo_E"] if args.quick else args.algorithms

    run_benchmark_suite(
        scenarios=scenarios,
        algorithms=algorithms,
        duration=dur,
        num_seeds=seeds,
    )


if __name__ == "__main__":
    main()
