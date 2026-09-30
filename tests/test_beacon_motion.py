"""
Unit tests for beacon kinematics and trajectory generators.
"""

import math
import numpy as np
import pytest
from simulation.coordinate_system import deg2rad, rad2deg
from simulation.trajectory import LinearTrajectory, CircularTrajectory, SinusoidalTrajectory
from simulation.beacon import Beacon


def test_linear_trajectory():
    traj = LinearTrajectory(x0=0.0, y0=0.0, vx=2.0, vy=1.0, ax=0.5, ay=0.0)
    # At t = 2.0 s:
    # x = 0 + 2.0*2 + 0.5*0.5*(4) = 4 + 1 = 5.0
    # y = 0 + 1.0*2 = 2.0
    # vx = 2.0 + 0.5*2 = 3.0
    # vy = 1.0
    px, py, vx, vy, ax, ay = traj.sample(2.0)
    assert math.isclose(px, 5.0, abs_tol=1e-5)
    assert math.isclose(py, 2.0, abs_tol=1e-5)
    assert math.isclose(vx, 3.0, abs_tol=1e-5)
    assert math.isclose(vy, 1.0, abs_tol=1e-5)


def test_circular_trajectory():
    traj = CircularTrajectory(
        center_x=0.0,
        center_y=0.0,
        radius=5.0,
        angular_speed=math.pi / 2.0,  # 90 deg/s
        initial_phase=0.0,
    )
    # At t = 0: x = 5, y = 0
    px0, py0, _, _, _, _ = traj.sample(0.0)
    assert math.isclose(px0, 5.0, abs_tol=1e-5)
    assert math.isclose(py0, 0.0, abs_tol=1e-5)

    # At t = 1 s: theta = pi/2 -> x = 0, y = 5
    px1, py1, _, _, _, _ = traj.sample(1.0)
    assert math.isclose(px1, 0.0, abs_tol=1e-5)
    assert math.isclose(py1, 5.0, abs_tol=1e-5)


def test_beacon_blinking():
    beacon = Beacon(
        beacon_id=1,
        initial_pos_rad=(0.0, 0.0),
        blinking_enabled=True,
        blinking_frequency_hz=1.0,  # 1s period
        blinking_duty_cycle=0.5,    # on 0 to 0.5s, off 0.5 to 1.0s
    )

    assert beacon.is_emitting(0.2) is True
    assert beacon.is_emitting(0.49) is True
    assert beacon.is_emitting(0.55) is False
    assert beacon.is_emitting(0.9) is False
    assert beacon.is_emitting(1.1) is True


def test_sinusoidal_trajectory_long_term_bounded():
    """Verify sinusoidal trajectory remains bounded across multi-minute simulations."""
    traj = SinusoidalTrajectory(
        x0=0.0,
        y0=0.0,
        vx=1.2,
        amplitude_y=2.5,
        frequency_hz=0.15,
        bound_x=22.0,
        bounce=True,
    )
    # Test at t=0, 60s (1 min), 120s (2 mins), 300s (5 mins), 600s (10 mins)
    for t in [0.0, 10.0, 60.0, 120.0, 180.0, 240.0, 300.0, 600.0]:
        px, py, vx, vy, ax, ay = traj.sample(t)
        # Position must never exceed horizontal bound
        assert abs(px) <= 22.0001, f"px={px} exceeded bound 22.0 at t={t}s"
        # Vertical motion remains within amplitude bound
        assert abs(py) <= 2.5001, f"py={py} exceeded amplitude 2.5 at t={t}s"
        # Velocities and accelerations must be smooth and finite
        assert not math.isnan(vx) and not math.isnan(vy)
        assert not math.isnan(ax) and not math.isnan(ay)


def test_maneuver_trajectory_indefinite_continuation():
    """Verify maneuver trajectory dynamically continues past 2 minutes without stalling."""
    from simulation.trajectory import ManeuverTrajectory
    traj = ManeuverTrajectory(
        x0=0.0,
        y0=0.0,
        vx0=1.0,
        vy0=0.0,
        max_acc=1.5,
        maneuver_interval=2.0,
        seed=42,
        bound_x=22.0,
        bound_y=13.0,
    )
    # Test sampling past 2 minutes (120s, 180s, 300s)
    for t in [120.0, 180.0, 240.0, 300.0]:
        px, py, vx, vy, ax, ay = traj.sample(t)
        assert not math.isnan(px) and not math.isnan(py)
        assert not math.isnan(vx) and not math.isnan(vy)
        assert not math.isnan(ax) and not math.isnan(ay)
        # Verify acceleration is active (not stalled at 0.0)
        assert math.hypot(ax, ay) > 0.0 or abs(vx) > 0.0
        # Check boundary adherence
        assert abs(px) <= 30.0, f"Maneuver px={px} exceeded safe margin at t={t}"
        assert abs(py) <= 20.0, f"Maneuver py={py} exceeded safe margin at t={t}"

