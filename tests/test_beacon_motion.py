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
