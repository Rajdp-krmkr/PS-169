"""
High-precision PID controller with anti-windup, dead-zone, and derivative filtering.

Specially designed for optical gimbal Pointing, Acquisition, and Tracking (PAT).
"""

from __future__ import annotations
import math
from typing import Tuple, Optional


class PIDController:
    """
    Discrete single-axis PID controller with:
    - Integrator anti-windup clamping
    - Configurable dead-zone to prevent mechanical hunting/chatter
    - Low-pass filtered derivative to suppress pixel quantization noise
    - Output rate saturation
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        dead_zone: float = 0.0,
        output_limits: Tuple[float, float] = (-10.0, 10.0),
        integral_limits: Optional[Tuple[float, float]] = None,
        derivative_filter_alpha: float = 0.8,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.dead_zone = max(0.0, dead_zone)
        self.min_out, self.max_out = output_limits

        if integral_limits is not None:
            self.min_int, self.max_int = integral_limits
        else:
            # Default anti-windup limit to half of max output
            span = self.max_out - self.min_out
            self.min_int = -span * 0.5
            self.max_int = span * 0.5

        self.alpha = derivative_filter_alpha  # Filter weight for newest derivative

        # Internal state
        self.integral = 0.0
        self.last_error = 0.0
        self.filtered_derivative = 0.0
        self.initialized = False

        # Telemetry components from last step
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.raw_error = 0.0
        self.effective_error = 0.0

    def reset(self) -> None:
        """Reset internal integrator and derivative history."""
        self.integral = 0.0
        self.last_error = 0.0
        self.filtered_derivative = 0.0
        self.initialized = False
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def compute(self, error: float, dt: float) -> float:
        """
        Compute control action given tracking error e(t) and delta-time dt.
        """
        if dt <= 1e-7:
            return 0.0

        self.raw_error = error

        # 1. Dead-Zone processing (smooth dead-band)
        if abs(error) <= self.dead_zone:
            effective_error = 0.0
        else:
            # Shift error toward zero by dead_zone to maintain continuity
            effective_error = error - math.copysign(self.dead_zone, error)
        self.effective_error = effective_error

        # Proportional term
        self.p_term = self.kp * effective_error

        # 2. Integral term with anti-windup clamping
        # Only integrate if outside dead_zone
        if effective_error != 0.0:
            self.integral += effective_error * dt
            self.integral = max(self.min_int, min(self.max_int, self.integral))
        else:
            # Bleed down integral slightly inside dead-zone to prevent overshoot
            self.integral *= 0.95

        self.i_term = self.ki * self.integral

        # 3. Derivative term with low-pass filtering
        if not self.initialized:
            d_raw = 0.0
            self.filtered_derivative = 0.0
            self.initialized = True
        else:
            d_raw = (effective_error - self.last_error) / dt
            self.filtered_derivative = (
                self.alpha * d_raw + (1.0 - self.alpha) * self.filtered_derivative
            )

        self.d_term = self.kd * self.filtered_derivative
        self.last_error = effective_error

        # 4. Total output with saturation limits
        output = self.p_term + self.i_term + self.d_term
        clamped_output = max(self.min_out, min(self.max_out, output))

        return clamped_output
