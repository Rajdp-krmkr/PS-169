"""
Kalman Filter for optical beacon state estimation and trajectory prediction.

Models 4D kinematic state: [azimuth, elevation, v_azimuth, v_elevation]^T
Supports:
- Discrete Wiener Process Acceleration (DWPA) process noise model
- Dynamic measurement noise updates
- Multi-step lookahead extrapolation for predictive lead-ahead control
- Chi-square innovation gating for outlier rejection
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np


class KalmanTracker:
    """
    Linear Kalman Filter for 2-axis optical angular tracking.
    """

    def __init__(
        self,
        q_process_spectral: float = 0.01,
        r_measurement_std: float = 0.002,  # ~0.11 deg standard deviation
        gate_threshold_sigma: float = 4.0,
    ) -> None:
        self.q_var = q_process_spectral
        self.r_std = r_measurement_std
        self.gate_threshold = gate_threshold_sigma**2  # Chi-square 2-DOF threshold

        # 4D State vector: [az, el, vx, vy]^T
        self.x = np.zeros(4, dtype=np.float64)
        # Covariance matrix
        self.P = np.eye(4, dtype=np.float64) * 1.0

        # Measurement matrix H: maps 4D state to 2D measurement [az, el]
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )

        # Default measurement covariance R
        self.R = np.eye(2, dtype=np.float64) * (self.r_std**2)

        self.initialized = False
        self.last_dt = 1.0 / 30.0

        # Innovation telemetry
        self.innovation = np.zeros(2, dtype=np.float64)
        self.innovation_cov = np.eye(2, dtype=np.float64)

    def initialize(self, az: float, el: float, p_init: float = 1.0) -> None:
        """Initialize filter state with first valid measurement."""
        self.x = np.array([az, el, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([p_init, p_init, p_init * 2.0, p_init * 2.0])
        self.initialized = True

    def reset(self) -> None:
        """Reset filter."""
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.initialized = False

    def predict(self, dt: float) -> np.ndarray:
        """
        Advance state and covariance estimate by dt.
        """
        if not self.initialized:
            return self.x.copy()

        self.last_dt = dt

        # State transition matrix F
        F = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        # Process noise covariance Q (DWPA model)
        dt2 = dt**2
        dt3 = dt**3
        q_scale = self.q_var
        Q = np.array(
            [[dt3 / 3.0, 0.0, dt2 / 2.0, 0.0],
             [0.0, dt3 / 3.0, 0.0, dt2 / 2.0],
             [dt2 / 2.0, 0.0, dt, 0.0],
             [0.0, dt2 / 2.0, 0.0, dt]],
            dtype=np.float64,
        ) * q_scale

        # Prior update
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        return self.x.copy()

    def update(
        self,
        measurement: Tuple[float, float],
        r_cov: Optional[np.ndarray] = None,
        use_gating: bool = True,
    ) -> bool:
        """
        Incorporate sensor measurement z = [az, el].
        Returns True if measurement passed gating and was fused.
        """
        z = np.array(measurement, dtype=np.float64)

        if not self.initialized:
            self.initialize(z[0], z[1])
            return True

        R = r_cov if r_cov is not None else self.R

        # Innovation: y = z - H * x_pred
        y = z - (self.H @ self.x)
        self.innovation = y

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + R
        self.innovation_cov = S

        # Chi-Square gating: d^2 = y^T * S^-1 * y
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            d2 = float(y.T @ S_inv @ y)

        if use_gating and d2 > self.gate_threshold:
            # Outlier rejected
            return False

        # Kalman Gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ S_inv

        # Posterior state update
        self.x = self.x + K @ y

        # Joseph form covariance update for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return True

    def predict_ahead(self, steps: int, dt: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Extrapolate future state H steps into the future.
        Returns: (pred_az, pred_el, pred_v_az, pred_v_el)
        """
        if not self.initialized:
            return float(self.x[0]), float(self.x[1]), 0.0, 0.0

        dt_val = dt if dt is not None else self.last_dt
        horizon_time = steps * dt_val

        # Linear velocity extrapolation: x(t + h) = x(t) + v * h
        pred_az = self.x[0] + self.x[2] * horizon_time
        pred_el = self.x[1] + self.x[3] * horizon_time

        return float(pred_az), float(pred_el), float(self.x[2]), float(self.x[3])

    @property
    def position(self) -> Tuple[float, float]:
        """Current estimated position (az, el)."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated velocity (v_az, v_el)."""
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        """Scalar angular speed in rad/s."""
        return math.hypot(float(self.x[2]), float(self.x[3]))
