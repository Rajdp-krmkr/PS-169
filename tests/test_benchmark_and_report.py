"""
Unit tests for benchmark execution and HTML report generation.
"""

import os
import shutil
import pytest
from evaluation.benchmark import run_single_benchmark_run, ALGORITHM_CONFIGS
from evaluation.report_generator import PerformanceReportGenerator


def test_single_benchmark_run():
    config = {
        "scenario_name": "TEST",
        "world": {"width_deg": 30.0, "height_deg": 20.0},
        "camera": {"resolution": [640, 480], "fov_horizontal_deg": 20.0, "fov_vertical_deg": 15.0, "fps": 30.0},
        "beacon": {
            "id": 1,
            "name": "test_beacon",
            "initial_pos_deg": [1.0, 0.5],
            "intensity": 255,
            "radius_pixels": 6.0,
            "trajectory": {"type": "linear", "vx_deg_s": 0.5, "vy_deg_s": 0.0},
        },
        "detection": {"binary_threshold": 140},
        "pid": {"kp_pan": 2.0, "kp_tilt": 2.0},
        "disturbances": {},
    }

    result = run_single_benchmark_run(config, algo_key="Algo_A", duration=1.0, seed=42)
    assert result["total_frames"] == 30
    assert "rmse_deg" in result
    assert "mean_error_deg" in result
    assert result["mean_error_deg"] < 2.0


def test_html_report_generation():
    test_rep_dir = "reports/test_report_gen"
    rep_gen = PerformanceReportGenerator(output_dir=test_rep_dir)

    try:
        metrics = {
            "rmse_deg": 0.45,
            "mean_error_deg": 0.38,
            "lock_retention_rate_pct": 82.5,
            "acquisition_time_sec": 0.25,
            "mean_detector_latency_ms": 5.2,
            "mean_fps": 30.0,
            "total_lost_events": 0,
        }
        telemetry = [
            {"sim_time": 0.0, "pointing_error_deg": 1.2, "camera_pan_deg": 0.0, "camera_tilt_deg": 0.0},
            {"sim_time": 0.5, "pointing_error_deg": 0.3, "camera_pan_deg": 0.8, "camera_tilt_deg": 0.1},
            {"sim_time": 1.0, "pointing_error_deg": 0.05, "camera_pan_deg": 1.5, "camera_tilt_deg": 0.2},
        ]
        benchmark_matrix = {
            "uav": {
                "Algo_A": {"algorithm_name": "Classical + PID", "rmse_mean": 0.85, "mean_error": 0.70, "lock_retention_mean": 60.0, "detector_latency_ms": 4.5, "total_loss_events": 0},
                "Algo_E": {"algorithm_name": "Hybrid + Pred + Kalman", "rmse_mean": 0.32, "mean_error": 0.25, "lock_retention_mean": 95.0, "detector_latency_ms": 6.8, "total_loss_events": 0},
            }
        }

        report_file = rep_gen.generate_report(
            scenario_name="UAV",
            metrics_summary=metrics,
            telemetry_records=telemetry,
            benchmark_matrix=benchmark_matrix,
        )

        assert os.path.exists(report_file)
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "SIH 2026 SMART INDIA HACKATHON" in content
            assert "0.450°" in content
            assert "data:image/png;base64" in content
    finally:
        if os.path.exists(test_rep_dir):
            shutil.rmtree(test_rep_dir)
