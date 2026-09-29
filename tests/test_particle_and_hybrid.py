"""
Unit tests for Particle Filter, Severity Calculator, and Adaptive Hybrid Tracking.
"""

import math
import numpy as np
import pytest

from tracking.particle import ParticleTracker
from tracking.severity import OpticalSeverityCalculator
from tracking.hybrid import HybridPATTracker


def test_particle_tracker():
    pf = ParticleTracker(num_particles=200, seed=42)
    pf.initialize(0.010, 0.005, vx=0.002, vy=-0.001)

    # 1. Prediction step
    prior = pf.predict(dt=0.033)
    assert prior.shape == (4,)
    assert math.isclose(prior[0], 0.010 + 0.002 * 0.033, abs_tol=0.005)

    # 2. Measurement update with high confidence
    post = pf.update(meas_az=0.0101, meas_el=0.0049, confidence=0.95)
    assert math.isclose(post[0], 0.0101, abs_tol=0.003)

    # 3. Measurement dropout (fading) -> coasts
    coast = pf.update(meas_az=None, meas_el=None, confidence=0.0)
    assert coast.shape == (4,)


def test_optical_severity_calculator():
    calc = OpticalSeverityCalculator()

    # Low error, high confidence -> low severity
    sev_calm = calc.compute(
        confidence=0.95,
        tracking_error_rad=0.0005,
        turbulence_level=0.05,
    )
    assert sev_calm < 0.30

    # Low confidence, high turbulence, high error -> severe degradation
    calc.reset()
    sev_bad = calc.compute(
        confidence=0.10,
        tracking_error_rad=0.015,
        turbulence_level=0.95,
        consecutive_misses=4,
    )
    assert sev_bad > 0.50


def test_hybrid_mode_switching_hysteresis():
    hybrid = HybridPATTracker(
        severity_switch_to_pf=0.50,
        severity_switch_to_kf=0.30,
        consecutive_calm_to_kf=4,
    )
    hybrid.initialize(0.01, 0.01)
    assert hybrid.active_mode == "KF"

    # Calm tracking in KF
    for _ in range(5):
        est = hybrid.step(0.033, 0.01, 0.01, confidence=0.95, turbulence_level=0.05)
    assert est.active_filter == "KF"

    # Surge severity (deep scintillation fade) -> switches to PF
    for i in range(1, 4):
        est = hybrid.step(
            0.033,
            meas_az=None,
            meas_el=None,
            confidence=0.0,
            turbulence_level=0.95,
            consecutive_misses=i,
        )
    assert est.active_filter == "PF"
    assert hybrid.total_switches >= 1

    # Return to calm -> requires severity decay and M=4 consecutive calm frames to switch back
    for f in range(2):
        est = hybrid.step(0.033, 0.01, 0.01, confidence=0.98, turbulence_level=0.0)
        assert est.active_filter == "PF"  # Still in hysteresis window

    # Additional calm frames to fulfill consecutive calm window
    for f in range(6):
        est = hybrid.step(0.033, 0.01, 0.01, confidence=0.98, turbulence_level=0.0)
    assert est.active_filter == "KF"
