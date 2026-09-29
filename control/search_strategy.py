"""
Coarse acquisition and reacquisition search strategies for FSOC PAT.

Implements:
- Raster Scan (Boustrophedon sweep)
- Spiral Scan (Archimedes spiral around current boresight)
- Predictive Search (Expanding trajectory search centered on Kalman extrapolation)
"""

from __future__ import annotations
import abc
import math
from typing import Tuple, Optional


class SearchStrategy(abc.ABC):
    """Abstract interface for gimbal scan and acquisition patterns."""

    @abc.abstractmethod
    def reset(self, center_pan_rad: float, center_tilt_rad: float) -> None:
        """Reset search scan centered at specified orientation."""
        pass

    @abc.abstractmethod
    def step(
        self, current_pan_rad: float, current_tilt_rad: float, dt: float
    ) -> Tuple[float, float]:
        """
        Compute command gimbal rates (cmd_pan_rate_rad_s, cmd_tilt_rate_rad_s).
        """
        pass


class SpiralScan(SearchStrategy):
    """
    Archimedes spiral scan expanding outward from an initial anchor pointing angle.
    r(t) = a * omega * t
    phi(t) = omega * t
    """

    def __init__(
        self,
        scan_radius_rad: float = 0.25,      # ~14.3 degrees max radius
        scan_frequency_hz: float = 0.8,     # 0.8 Hz rotation
        expansion_rate: float = 0.02,       # Expansion per radian of rotation
    ) -> None:
        self.max_radius = scan_radius_rad
        self.omega = 2.0 * math.pi * scan_frequency_hz
        self.expansion_coeff = expansion_rate

        self.origin_pan = 0.0
        self.origin_tilt = 0.0
        self.elapsed_time = 0.0

    def reset(self, center_pan_rad: float, center_tilt_rad: float) -> None:
        self.origin_pan = center_pan_rad
        self.origin_tilt = center_tilt_rad
        self.elapsed_time = 0.0

    def step(
        self, current_pan_rad: float, current_tilt_rad: float, dt: float
    ) -> Tuple[float, float]:
        self.elapsed_time += dt

        # Current spiral phase
        theta = self.omega * self.elapsed_time
        r = min(self.max_radius, self.expansion_coeff * theta)

        # Desired position
        target_pan = self.origin_pan + r * math.cos(theta)
        target_tilt = self.origin_tilt + r * math.sin(theta)

        # Rate command towards target position (proportional steering)
        kp_track = 8.0
        rate_pan = kp_track * (target_pan - current_pan_rad)
        rate_tilt = kp_track * (target_tilt - current_tilt_rad)

        return rate_pan, rate_tilt


class RasterScan(SearchStrategy):
    """
    Boustrophedon / Raster scan sweeping horizontally and stepping vertically.
    """

    def __init__(
        self,
        pan_range_rad: float = 0.50,         # Total azimuth sweep span
        tilt_range_rad: float = 0.30,        # Total elevation sweep span
        scan_speed_rad_s: float = 0.15,      # Azimuth sweep velocity
        elevation_step_rad: float = 0.08,    # Step per line (~70% of FOV height)
    ) -> None:
        self.pan_span = pan_range_rad
        self.tilt_span = tilt_range_rad
        self.scan_speed = scan_speed_rad_s
        self.el_step = elevation_step_rad

        self.center_pan = 0.0
        self.center_tilt = 0.0

        self.min_pan = 0.0
        self.max_pan = 0.0
        self.min_tilt = 0.0
        self.max_tilt = 0.0

        self.sweep_direction = 1.0  # +1 right, -1 left
        self.target_tilt = 0.0
        self.target_pan = 0.0

    def reset(self, center_pan_rad: float, center_tilt_rad: float) -> None:
        self.center_pan = center_pan_rad
        self.center_tilt = center_tilt_rad

        self.min_pan = center_pan_rad - self.pan_span / 2.0
        self.max_pan = center_pan_rad + self.pan_span / 2.0
        self.min_tilt = center_tilt_rad - self.tilt_span / 2.0
        self.max_tilt = center_tilt_rad + self.tilt_span / 2.0

        self.target_pan = self.min_pan
        self.target_tilt = self.max_tilt
        self.sweep_direction = 1.0

    def step(
        self, current_pan_rad: float, current_tilt_rad: float, dt: float
    ) -> Tuple[float, float]:
        # Move target_pan along sweep direction
        self.target_pan += self.sweep_direction * self.scan_speed * dt

        # Check line edge
        if self.sweep_direction > 0 and self.target_pan >= self.max_pan:
            self.target_pan = self.max_pan
            self.sweep_direction = -1.0
            self.target_tilt -= self.el_step
            if self.target_tilt < self.min_tilt:
                self.target_tilt = self.max_tilt  # Wrap scan to top
        elif self.sweep_direction < 0 and self.target_pan <= self.min_pan:
            self.target_pan = self.min_pan
            self.sweep_direction = 1.0
            self.target_tilt -= self.el_step
            if self.target_tilt < self.min_tilt:
                self.target_tilt = self.max_tilt

        kp = 6.0
        rate_pan = kp * (self.target_pan - current_pan_rad)
        rate_tilt = kp * (self.target_tilt - current_tilt_rad)

        return rate_pan, rate_tilt


class PredictiveSearch(SearchStrategy):
    """
    Adaptive search centered on Kalman extrapolated target trajectory cone.
    Combines forward dead-reckoning motion with small oscillatory search dither.
    """

    def __init__(
        self,
        predicted_pos_rad: Tuple[float, float] = (0.0, 0.0),
        predicted_vel_rad_s: Tuple[float, float] = (0.0, 0.0),
        search_radius_rad: float = 0.08,
        search_freq_hz: float = 1.2,
    ) -> None:
        self.pos = list(predicted_pos_rad)
        self.vel = list(predicted_vel_rad_s)
        self.radius = search_radius_rad
        self.omega = 2.0 * math.pi * search_freq_hz
        self.elapsed = 0.0

    def set_prediction(
        self, pos_rad: Tuple[float, float], vel_rad_s: Tuple[float, float]
    ) -> None:
        """Update seed trajectory from Kalman filter."""
        self.pos = [pos_rad[0], pos_rad[1]]
        self.vel = [vel_rad_s[0], vel_rad_s[1]]
        self.elapsed = 0.0

    def reset(self, center_pan_rad: float, center_tilt_rad: float) -> None:
        self.pos = [center_pan_rad, center_tilt_rad]
        self.elapsed = 0.0

    def step(
        self, current_pan_rad: float, current_tilt_rad: float, dt: float
    ) -> Tuple[float, float]:
        self.elapsed += dt

        # Forward dead-reckoning
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt

        # Search dither
        dither_az = self.radius * math.cos(self.omega * self.elapsed)
        dither_el = self.radius * math.sin(self.omega * self.elapsed)

        target_pan = self.pos[0] + dither_az
        target_tilt = self.pos[1] + dither_el

        kp = 8.0
        rate_pan = kp * (target_pan - current_pan_rad) + self.vel[0]
        rate_tilt = kp * (target_tilt - current_tilt_rad) + self.vel[1]

        return rate_pan, rate_tilt
