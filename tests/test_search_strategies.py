"""
Unit tests for raster, spiral, and predictive search patterns.
"""

import math
import pytest
from control.search_strategy import SpiralScan, RasterScan, PredictiveSearch


def test_spiral_scan_expansion():
    scan = SpiralScan(scan_radius_rad=0.2, scan_frequency_hz=1.0)
    scan.reset(center_pan_rad=0.0, center_tilt_rad=0.0)

    # Initial step at t=0 has zero rate
    r_pan0, r_tilt0 = scan.step(0.0, 0.0, dt=0.01)
    # Over time, rate commands steer outward in spiral
    rates = []
    for _ in range(50):
        rp, rt = scan.step(0.0, 0.0, dt=0.02)
        rates.append(math.hypot(rp, rt))

    assert max(rates) > 0.05


def test_raster_scan_bounds():
    scan = RasterScan(pan_range_rad=0.4, tilt_range_rad=0.2, scan_speed_rad_s=0.5)
    scan.reset(center_pan_rad=0.0, center_tilt_rad=0.0)

    # Verify bounds initialized correctly
    assert math.isclose(scan.min_pan, -0.2, abs_tol=1e-5)
    assert math.isclose(scan.max_pan, 0.2, abs_tol=1e-5)

    # Step through several sweeps and ensure lines sweep back and forth
    directions = set()
    for _ in range(60):
        scan.step(current_pan_rad=scan.target_pan, current_tilt_rad=scan.target_tilt, dt=0.05)
        directions.add(scan.sweep_direction)

    assert 1.0 in directions
    assert -1.0 in directions


def test_predictive_search_motion():
    pred_search = PredictiveSearch(
        predicted_pos_rad=(0.1, 0.05),
        predicted_vel_rad_s=(0.02, -0.01),
        search_radius_rad=0.05,
    )
    # Step forward in time
    rate_p, rate_t = pred_search.step(current_pan_rad=0.1, current_tilt_rad=0.05, dt=0.1)
    # Rate should incorporate target velocity feed-forward
    assert abs(rate_p) > 0.01
