"""
Unit tests for Dynamic Safe FOV Engine, Loss-of-Lock Risk Engine,
and Fine-PAT Handoff Readiness Engine.
"""

import math
import pytest

from simulation.camera import VirtualCamera
from control.safe_fov import DynamicSafeFOVEngine
from control.handoff import FinePATHandoffEngine


def test_dynamic_safe_fov_computation():
    camera = VirtualCamera(resolution=(1280, 720), fov_horizontal_deg=20.0, fov_vertical_deg=11.25)
    engine = DynamicSafeFOVEngine(camera=camera)

    # 1. Target centered with low speed and zero vibration -> STABLE
    telem = engine.compute(
        rel_az_rad=0.0,
        rel_el_rad=0.0,
        target_vel_rad_s=(0.0, 0.0),
        vibration_amp_rad=0.0,
    )
    assert telem.risk_state == "STABLE"
    assert telem.risk_score <= 0.30
    assert telem.dist_to_safe_edge_mrad > 50.0

    # 2. Target near safe boundary with high velocity -> LOCK AT RISK or CRITICAL
    half_fov = camera.fov_horizontal_rad / 2.0
    telem_edge = engine.compute(
        rel_az_rad=half_fov * 0.95,
        rel_el_rad=0.0,
        target_vel_rad_s=(0.02, 0.0),  # 20 mrad/s
        vibration_amp_rad=0.001,
    )
    assert telem_edge.risk_state in ("LOCK AT RISK", "CRITICAL")
    assert telem_edge.risk_score >= 0.70

    # 3. Target physically occluded -> CRITICAL
    telem_occ = engine.compute(
        rel_az_rad=0.0,
        rel_el_rad=0.0,
        is_occluded=True,
    )
    assert telem_occ.risk_state == "CRITICAL"
    assert telem_occ.risk_score >= 0.95
    assert "occlusion" in telem_occ.diagnostic_message.lower()


def test_fine_pat_handoff_engine():
    handoff = FinePATHandoffEngine(window_size=10, alignment_threshold_mrad=2.0)

    # Initially with zero/no history -> NOT READY
    res_init = handoff.update(
        tracking_error_mrad=5.0,
        is_locked_or_tracking=False,
    )
    assert res_init.handoff_state == "NOT READY"
    assert res_init.is_ready is False

    # Simulate 12 consecutive frames with very small error (0.5 mrad) and locked state
    for _ in range(12):
        res = handoff.update(
            tracking_error_mrad=0.5,
            is_locked_or_tracking=True,
            target_vel_rad_s=(0.001, 0.0),
            risk_state="STABLE",
        )

    # After settling into sub-milliradian alignment -> READY
    assert res.handoff_score_pct >= 85.0
    assert res.handoff_state == "READY"
    assert res.is_ready is True
    assert res.rolling_rms_error_mrad < 1.0
