"""
Fine-PAT Handoff Readiness Engine for FSOC Optical Links.

In Free-Space Optical Communication terminals, coarse gimbal PAT must align
the optical beam to within an angular tolerance before handing off the link
to the Fine-PAT subsystem (Fast Steering Mirrors [FSM] or piezo fine deflectors).

Evaluates:
- Instantaneous radial alignment error (mrad).
- Sliding-window RMS jitter over the past N frames.
- Gimbal and target residual angular velocity norm.
- Lock persistence rate across the evaluation window.
- Proximity to Safe FOV boundary.

Outputs:
- Multi-factor weighted Handoff Readiness Score [0.0% to 100.0%].
- Handoff Readiness State:
  - READY: Fine-PAT can immediately acquire and engage optical link.
  - STABILIZING: Error is damping down toward handoff threshold.
  - NOT READY: Excessive error, residual angular velocity, or unlocked state.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class HandoffTelemetry:
    handoff_score_pct: float
    handoff_state: str  # "READY", "STABILIZING", "NOT READY"
    instantaneous_error_mrad: float
    rolling_rms_error_mrad: float
    is_ready: bool
    diagnostic_summary: str


class FinePATHandoffEngine:
    """
    Real-time multi-criteria evaluator for Coarse-to-Fine PAT link handover.
    """

    def __init__(
        self,
        window_size: int = 15,
        alignment_threshold_mrad: float = 3.5,
        rms_threshold_mrad: float = 2.5,
        max_speed_norm_mrad_s: float = 45.0,
    ) -> None:
        self.window_size = window_size
        self.alignment_thresh = alignment_threshold_mrad
        self.rms_thresh = rms_threshold_mrad
        self.max_speed_norm = max_speed_norm_mrad_s

        self.error_history: deque[float] = deque(maxlen=window_size)
        self.lock_history: deque[bool] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Clear error and lock history."""
        self.error_history.clear()
        self.lock_history.clear()

    def update(
        self,
        tracking_error_mrad: float,
        is_locked_or_tracking: bool,
        target_vel_rad_s: tuple[float, float] = (0.0, 0.0),
        risk_state: str = "STABLE",
    ) -> HandoffTelemetry:
        """
        Ingest current frame metrics and compute Fine-PAT handoff readiness.
        """
        self.error_history.append(float(tracking_error_mrad))
        self.lock_history.append(bool(is_locked_or_tracking))

        # 1. Rolling RMS Error Calculation
        if len(self.error_history) > 0:
            rms_err = math.sqrt(sum(e**2 for e in self.error_history) / len(self.error_history))
        else:
            rms_err = tracking_error_mrad

        # 2. Multi-factor criteria scoring [0.0 to 1.0]
        # Factor A: Current radial alignment (Weight: 30%)
        # Full score when error <= 0.5 mrad, scaling down to threshold
        if tracking_error_mrad <= 0.5:
            c_align = 1.0
        else:
            c_align = max(0.0, min(1.0, 1.0 - (tracking_error_mrad - 0.5) / max(0.1, self.alignment_thresh - 0.5)))

        # Factor B: Rolling RMS jitter stability (Weight: 25%)
        if rms_err <= 0.5:
            c_rms = 1.0
        else:
            c_rms = max(0.0, min(1.0, 1.0 - (rms_err - 0.5) / max(0.1, self.rms_thresh - 0.5)))

        # Factor C: Lock persistence rate (Weight: 20%)
        lock_rate = sum(self.lock_history) / len(self.lock_history) if self.lock_history else 0.0
        c_lock = lock_rate

        # Factor D: Residual angular velocity limit (Weight: 15%)
        speed_norm_mrad_s = math.hypot(target_vel_rad_s[0], target_vel_rad_s[1]) * 1e3
        c_speed = max(0.0, min(1.0, 1.0 - (speed_norm_mrad_s / self.max_speed_norm)))

        # Factor E: Safe FOV margin status (Weight: 10%)
        c_margin = 1.0 if risk_state == "STABLE" else 0.5 if risk_state == "WARNING" else 0.0

        # Weighted aggregate score
        raw_score = (
            0.30 * c_align +
            0.25 * c_rms +
            0.20 * c_lock +
            0.15 * c_speed +
            0.10 * c_margin
        )
        handoff_score_pct = round(max(0.0, min(100.0, raw_score * 100.0)), 1)

        # Categorization
        if handoff_score_pct >= 85.0 and is_locked_or_tracking and tracking_error_mrad <= self.alignment_thresh:
            handoff_state = "READY"
            is_ready = True
            diag = (
                f"Fine-PAT Handoff READY ({handoff_score_pct}%). "
                f"Error: {tracking_error_mrad:.2f} mrad, RMS: {rms_err:.2f} mrad."
            )
        elif handoff_score_pct >= 50.0:
            handoff_state = "STABILIZING"
            is_ready = False
            diag = (
                f"Fine-PAT Handoff STABILIZING ({handoff_score_pct}%). "
                f"Damping residual pointing jitter (RMS: {rms_err:.2f} mrad)."
            )
        else:
            handoff_state = "NOT READY"
            is_ready = False
            diag = (
                f"Fine-PAT Handoff NOT READY ({handoff_score_pct}%). "
                f"Alignment error {tracking_error_mrad:.2f} mrad exceeds threshold."
            )

        return HandoffTelemetry(
            handoff_score_pct=handoff_score_pct,
            handoff_state=handoff_state,
            instantaneous_error_mrad=round(tracking_error_mrad, 3),
            rolling_rms_error_mrad=round(rms_err, 3),
            is_ready=is_ready,
            diagnostic_summary=diag,
        )
