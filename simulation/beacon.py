"""
Optical beacon target entity for FSOC coarse alignment simulation.

Models:
- Kinematic state (position, velocity, acceleration in angular world coordinates)
- Point Spread Function (PSF) visual rendering (Airy disc / Gaussian beam profile)
- Temporal blinking signature (frequency, duty cycle) for signature verification
- Distractor vs primary beacon classification
"""

from __future__ import annotations
import collections
import math
from typing import Tuple, List, Optional
import numpy as np
import cv2

from simulation.coordinate_system import deg2rad
from simulation.trajectory import Trajectory


class Beacon:
    """
    Simulated optical beacon or optical distractor.
    """

    def __init__(
        self,
        beacon_id: int,
        name: str = "beacon",
        is_primary: bool = True,
        initial_pos_rad: Tuple[float, float] = (0.0, 0.0),
        initial_vel_rad_s: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 255.0,
        radius_pixels: float = 6.0,
        color_bgr: Tuple[int, int, int] = (255, 255, 255),
        trajectory: Optional[Trajectory] = None,
        blinking_enabled: bool = False,
        blinking_frequency_hz: float = 2.0,
        blinking_duty_cycle: float = 0.5,
    ) -> None:
        self.id = beacon_id
        self.name = name
        self.is_primary = is_primary

        self.pos = np.array(initial_pos_rad, dtype=np.float64)  # [az, el] in rad
        self.vel = np.array(initial_vel_rad_s, dtype=np.float64)  # [d_az, d_el] in rad/s
        self.acc = np.zeros(2, dtype=np.float64)

        self.intensity = float(intensity)
        self.radius_pixels = float(radius_pixels)
        self.color_bgr = color_bgr

        self.trajectory = trajectory
        self.blinking_enabled = blinking_enabled
        self.blinking_freq = blinking_frequency_hz
        self.blinking_duty = blinking_duty_cycle

        self.active = True
        self.occluded = False

        # State history for visualization trail
        self.history: collections.deque[Tuple[float, float]] = collections.deque(maxlen=100)

    def update(self, t: float, dt: float) -> None:
        """Advance beacon state to time t."""
        if not self.active:
            return

        if self.trajectory is not None:
            px, py, vx, vy, ax, ay = self.trajectory.sample(t)
            # If trajectory produces degrees, convert to radians
            self.pos[0] = deg2rad(px)
            self.pos[1] = deg2rad(py)
            self.vel[0] = deg2rad(vx)
            self.vel[1] = deg2rad(vy)
            self.acc[0] = deg2rad(ax)
            self.acc[1] = deg2rad(ay)
        else:
            # Standard ballistic Euler integration
            self.pos += self.vel * dt + 0.5 * self.acc * (dt**2)
            self.vel += self.acc * dt

        self.history.append((float(self.pos[0]), float(self.pos[1])))

    def is_emitting(self, t: float) -> bool:
        """Determine if beacon is optical-emitting at time t (blinking modulation)."""
        if not self.active or self.occluded:
            return False
        if not self.blinking_enabled:
            return True

        period = 1.0 / max(self.blinking_freq, 1e-4)
        phase_in_period = (t % period) / period
        return phase_in_period <= self.blinking_duty

    def render_to_frame(
        self,
        frame: np.ndarray,
        pixel_u: float,
        pixel_v: float,
        intensity_factor: float = 1.0,
    ) -> None:
        """
        Render sub-pixel optical Gaussian point spread function (PSF) onto camera frame.
        """
        h, w = frame.shape[:2]
        if not (0 <= pixel_u < w and 0 <= pixel_v < h):
            return

        # Gaussian spot profile
        sigma = max(self.radius_pixels / 2.5, 0.8)
        kernel_radius = int(math.ceil(sigma * 3.5))

        u_min = max(0, int(math.floor(pixel_u - kernel_radius)))
        u_max = min(w, int(math.ceil(pixel_u + kernel_radius + 1)))
        v_min = max(0, int(math.floor(pixel_v - kernel_radius)))
        v_max = min(h, int(math.ceil(pixel_v + kernel_radius + 1)))

        if u_min >= u_max or v_min >= v_max:
            return

        grid_y, grid_x = np.ogrid[v_min:v_max, u_min:u_max]
        dist_sq = (grid_x - pixel_u) ** 2 + (grid_y - pixel_v) ** 2
        psf = np.exp(-0.5 * dist_sq / (sigma**2))

        # Peak intensity modulated by color
        peak = self.intensity * intensity_factor
        color_arr = np.array(self.color_bgr, dtype=np.float32) / 255.0

        for c in range(3):
            spot = psf * peak * color_arr[c]
            # Additive blending with saturation clamping
            cur_roi = frame[v_min:v_max, u_min:u_max, c].astype(np.float32)
            frame[v_min:v_max, u_min:u_max, c] = np.clip(cur_roi + spot, 0, 255).astype(
                np.uint8
            )
