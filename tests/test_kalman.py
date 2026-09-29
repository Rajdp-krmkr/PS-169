"""
Unit tests for linear Kalman filter tracking and trajectory prediction.
"""

import math
import numpy as np
import pytest
from tracking.kalman import KalmanTracker


def test_kalman_initialization():
    tracker = KalmanTracker()
    assert tracker.initialized is False
    tracker.initialize(az=0.5, el=-0.2)
    assert tracker.initialized is True
    assert tracker.position == (0.5, -0.2)
    assert tracker.velocity == (0.0, 0.0)


def test_kalman_noise_smoothing():
    tracker = KalmanTracker(q_process_spectral=0.1, r_measurement_std=0.05)
    tracker.initialize(0.0, 0.0)

    # Stationary target at (0.0, 0.0) with Gaussian noise
    rng = np.random.default_rng(42)
    raw_measurements = []
    estimates = []

    dt = 1.0 / 30.0
    for _ in range(50):
        tracker.predict(dt)
        meas_az = float(rng.normal(0.0, 0.05))
        meas_el = float(rng.normal(0.0, 0.05))
        raw_measurements.append(meas_az)
        tracker.update((meas_az, meas_el))
        estimates.append(tracker.position[0])

    # In stationary conditions, Kalman estimate variance should be lower than measurement variance
    raw_var = np.var(raw_measurements[15:])
    est_var = np.var(estimates[15:])
    assert est_var < raw_var * 0.7


def test_kalman_velocity_estimation():
    tracker = KalmanTracker(q_process_spectral=0.005, r_measurement_std=0.002)
    tracker.initialize(0.0, 0.0)

    true_vx = 0.05  # rad/s (~2.86 deg/s)
    true_vy = -0.02 # rad/s
    dt = 1.0 / 30.0

    cur_x = 0.0
    cur_y = 0.0
    rng = np.random.default_rng(42)

    for _ in range(120):
        cur_x += true_vx * dt
        cur_y += true_vy * dt
        tracker.predict(dt)
        noisy_x = cur_x + float(rng.normal(0.0, 0.001))
        noisy_y = cur_y + float(rng.normal(0.0, 0.001))
        tracker.update((noisy_x, noisy_y))

    est_vx, est_vy = tracker.velocity
    assert math.isclose(est_vx, true_vx, abs_tol=0.01)
    assert math.isclose(est_vy, true_vy, abs_tol=0.01)


def test_kalman_lookahead_prediction():
    tracker = KalmanTracker()
    tracker.initialize(0.0, 0.0)
    # Manually set state: pos = (1.0, 2.0), vel = (0.5, -0.2)
    tracker.x = np.array([1.0, 2.0, 0.5, -0.2], dtype=np.float64)

    # Predict 10 steps ahead (dt = 0.1s -> horizon = 1.0s)
    # Expected: x = 1.0 + 0.5*1.0 = 1.5, y = 2.0 - 0.2*1.0 = 1.8
    p_az, p_el, p_vx, p_vy = tracker.predict_ahead(steps=10, dt=0.1)
    assert math.isclose(p_az, 1.5, abs_tol=1e-5)
    assert math.isclose(p_el, 1.8, abs_tol=1e-5)
    assert math.isclose(p_vx, 0.5, abs_tol=1e-5)
    assert math.isclose(p_vy, -0.2, abs_tol=1e-5)
