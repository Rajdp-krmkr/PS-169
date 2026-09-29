"""
Optical Disturbance & Scintillation Severity Calculator for Adaptive Tracking.

Computes a normalized real-time severity score S in [0.0, 1.0] quantifying
the combined degradation of the optical tracking channel:
- Signal-to-Noise Ratio (SNR) and detector confidence dropouts
- Normalized innovation / tracking error surges
- Atmospheric turbulence intensity (Kolmogorov scintillation)
- High-dynamic angular acceleration maneuvers

Used by HybridPATTracker to determine optimal estimation regime:
- Calm conditions (S < 0.30) -> High-rate Kalman Filter (optimal Gaussian estimator)
- Severe scintillation/fading (S > 0.55) -> Particle Filter (handles multi-modal non-Gaussian fades)
"""

from __future__ import annotations
import math
from typing import Optional, Tuple


class OpticalSeverityCalculator:
    """
    Computes real-time optical tracking severity for filter mode switching.
    """

    def __init__(
        self,
        error_scale_rad: float = 0.005,  # ~0.28 deg
        accel_scale_rad_s2: float = 0.05,
    ) -> None:
        self.error_scale = error_scale_rad
        self.accel_scale = accel_scale_rad_s2
        self.smoothed_severity: float = 0.0

    def compute(
        self,
        confidence: float,
        tracking_error_rad: float,
        turbulence_level: float = 0.0,
        is_occluded: bool = False,
        consecutive_misses: int = 0,
        alpha: float = 0.4,
    ) -> float:
        """
        Evaluate instantaneous channel severity and apply exponential moving average.

        Args:
            confidence: Detector confidence in [0.0, 1.0]
            tracking_error_rad: Current radial tracking error (rad)
            turbulence_level: Normalized turbulence intensity [0.0, 1.0]
            is_occluded: Whether line of sight is obstructed
            consecutive_misses: Count of successive undetected frames
            alpha: EMA smoothing coefficient
        """
        if is_occluded:
            inst_sev = 1.0
        else:
            # Component 1: Confidence loss (Weight: 40%)
            c_loss = max(0.0, min(1.0, 1.0 - confidence))

            # Component 2: Tracking error surge (Weight: 30%)
            c_err = max(0.0, min(1.0, tracking_error_rad / self.error_scale))

            # Component 3: Turbulence scintillation (Weight: 20%)
            c_turb = max(0.0, min(1.0, turbulence_level))

            # Component 4: Consecutive dropouts (Weight: 10%)
            c_drop = max(0.0, min(1.0, consecutive_misses / 4.0))

            inst_sev = 0.40 * c_loss + 0.30 * c_err + 0.20 * c_turb + 0.10 * c_drop

        if not hasattr(self, "_initialized") or not self._initialized:
            self.smoothed_severity = inst_sev
            self._initialized = True
        else:
            self.smoothed_severity = (1.0 - alpha) * self.smoothed_severity + alpha * inst_sev

        return round(float(self.smoothed_severity), 3)

    def reset(self) -> None:
        self.smoothed_severity = 0.0
        self._initialized = False
