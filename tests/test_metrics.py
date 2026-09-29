"""
Unit tests for performance metrics calculator.
"""

import math
import pytest
from evaluation.metrics import PerformanceMetricsCalculator, FrameTelemetrySample


def test_metrics_calculation():
    calc = PerformanceMetricsCalculator(lock_threshold_deg=0.08)

    # Record 10 synthetic frames
    # Frame 0 to 4: not locked, error decreases 0.20 -> 0.08
    # Frame 5 to 9: locked, error 0.05
    for i in range(10):
        t = i * 0.1
        err = 0.20 - i * 0.02 if i < 5 else 0.05
        locked = (err <= 0.08)
        calc.record_sample(
            FrameTelemetrySample(
                sim_time=t,
                ground_truth_az_deg=1.0,
                ground_truth_el_deg=0.5,
                camera_pan_deg=1.0 - err,
                camera_tilt_deg=0.5,
                pointing_error_deg=err,
                is_in_fov=True,
                is_locked=locked,
                pat_state="TRACKING" if locked else "ACQUIRING",
                detector_latency_ms=5.0,
                fps=30.0,
            )
        )

    summary = calc.compute_summary()
    assert summary.total_frames == 10
    assert summary.initial_error_deg == 0.20
    assert summary.final_error_deg == 0.05
    assert summary.acquisition_time_sec == 0.5  # First locked at frame 5 (t=0.5s)
    assert summary.lock_retention_rate_pct == 50.0  # 5 out of 10 frames locked
    assert summary.mean_detector_latency_ms == 5.0
    assert summary.rmse_deg > 0.05
