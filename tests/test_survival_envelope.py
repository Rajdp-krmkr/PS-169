"""
Unit tests for Tracking Survival Envelope Matrix and PAT-RED Adversarial Discovery.
"""

import pytest
from evaluation.survival_envelope import evaluate_survival_envelope, run_pat_red_team_discovery


def test_survival_envelope_sweep():
    # Run short evaluation across matrix
    res = evaluate_survival_envelope(num_frames=15, dt=0.033)
    assert len(res.velocities_mrad_s) == 5
    assert len(res.turbulences_label) == 3
    assert len(res.lock_retention_matrix_pct) == 5
    assert len(res.lock_retention_matrix_pct[0]) == 3
    # Check that low velocity under low turbulence has high retention
    assert res.lock_retention_matrix_pct[0][0] >= 80.0
    assert "Safe Tracking Boundary" in res.safe_boundary_summary


def test_pat_red_team_discovery():
    res = run_pat_red_team_discovery(seed=42)
    assert res.breaking_point_found is True
    assert res.failure_id.startswith("PATRED-")
    assert res.target_turn_rate_mrad_s >= 20.0
    assert res.lock_retention_pct < 90.0  # Must violate 90% requirement
    assert len(res.root_cause_diagnosis) > 20
