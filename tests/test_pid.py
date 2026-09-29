"""
Unit tests for PID controller with anti-windup, dead-zone, and saturation.
"""

import math
import pytest
from control.pid import PIDController


def test_pid_proportional():
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, dead_zone=0.0, output_limits=(-10.0, 10.0))
    out = pid.compute(error=1.5, dt=0.1)
    assert math.isclose(out, 3.0, abs_tol=1e-5)


def test_pid_dead_zone():
    pid = PIDController(kp=2.0, ki=0.0, kd=0.0, dead_zone=0.5, output_limits=(-10.0, 10.0))
    # Error within dead zone produces zero
    assert pid.compute(error=0.3, dt=0.1) == 0.0
    assert pid.compute(error=-0.4, dt=0.1) == 0.0

    # Error outside dead zone has smooth dead-band reduction (1.5 - 0.5 = 1.0 -> out = 2.0)
    out = pid.compute(error=1.5, dt=0.1)
    assert math.isclose(out, 2.0, abs_tol=1e-5)


def test_pid_anti_windup():
    pid = PIDController(
        kp=0.0,
        ki=1.0,
        kd=0.0,
        integral_limits=(-2.0, 2.0),
        output_limits=(-5.0, 5.0),
    )

    # Accumulate massive error
    for _ in range(50):
        pid.compute(error=10.0, dt=0.1)

    assert pid.integral <= 2.0
    assert pid.integral >= -2.0


def test_pid_saturation():
    pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_limits=(-5.0, 5.0))
    out = pid.compute(error=2.0, dt=0.1)
    assert math.isclose(out, 5.0, abs_tol=1e-5)
