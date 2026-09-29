"""
Virtual Pan-Tilt Camera model for FSOC coarse alignment.

Simulates:
- 2-Axis Gimbal kinematics (Pan / Azimuth, Tilt / Elevation)
- Velocity and physical angular travel limits
- Optical projection and FOV culling
- Synthesized camera sensor frame rendering with sub-pixel PSF
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np

from simulation.coordinate_system import OpticalProjection, deg2rad, rad2deg, normalize_angle_rad
from simulation.beacon import Beacon


class VirtualCamera:
    """
    Simulated optical sensor mounted on a 2-axis pan-tilt gimbal mechanism.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        fov_horizontal_deg: float = 20.0,
        fov_vertical_deg: float = 11.25,
        pan_limits_deg: Tuple[float, float] = (-60.0, 60.0),
        tilt_limits_deg: Tuple[float, float] = (-35.0, 35.0),
        max_pan_speed_deg_s: float = 25.0,
        max_tilt_speed_deg_s: float = 25.0,
        fps: float = 30.0,
    ) -> None:
        self.resolution = resolution
        self.width, self.height = resolution

        self.fov_h_rad = deg2rad(fov_horizontal_deg)
        self.fov_v_rad = deg2rad(fov_vertical_deg)

        self.pan_limits = (deg2rad(pan_limits_deg[0]), deg2rad(pan_limits_deg[1]))
        self.tilt_limits = (deg2rad(tilt_limits_deg[0]), deg2rad(tilt_limits_deg[1]))

        self.max_pan_speed = deg2rad(max_pan_speed_deg_s)
        self.max_tilt_speed = deg2rad(max_tilt_speed_deg_s)
        self.fps = fps

        # Orientation in radians
        self.pan: float = 0.0
        self.tilt: float = 0.0

        # Current gimbal rates (rad/s)
        self.pan_velocity: float = 0.0
        self.tilt_velocity: float = 0.0

        # Optical projection helper
        self.projection = OpticalProjection(
            resolution=resolution,
            fov_horizontal_rad=self.fov_h_rad,
            fov_vertical_rad=self.fov_v_rad,
        )

        # Disturbance offsets (injected by disturbance engine)
        self.pan_jitter_offset: float = 0.0
        self.tilt_jitter_offset: float = 0.0

    @property
    def fov_horizontal_rad(self) -> float:
        return self.fov_h_rad

    @property
    def fov_vertical_rad(self) -> float:
        return self.fov_v_rad

    @property
    def effective_pan(self) -> float:
        """Effective optical pointing azimuth including platform vibration/jitter."""
        return self.pan + self.pan_jitter_offset

    @property
    def effective_tilt(self) -> float:
        """Effective optical pointing elevation including platform vibration/jitter."""
        return self.tilt + self.tilt_jitter_offset

    def set_command_rate(self, pan_rate_rad_s: float, tilt_rate_rad_s: float) -> None:
        """Command gimbal angular rates with hard saturation limits."""
        self.pan_velocity = float(
            np.clip(pan_rate_rad_s, -self.max_pan_speed, self.max_pan_speed)
        )
        self.tilt_velocity = float(
            np.clip(tilt_rate_rad_s, -self.max_tilt_speed, self.max_tilt_speed)
        )

    def set_orientation(self, pan_rad: float, tilt_rad: float) -> None:
        """Directly position camera gimbal within travel limits."""
        self.pan = float(np.clip(pan_rad, self.pan_limits[0], self.pan_limits[1]))
        self.tilt = float(np.clip(tilt_rad, self.tilt_limits[0], self.tilt_limits[1]))

    def step(self, dt: float) -> None:
        """Integrate gimbal kinematics by time-step dt."""
        self.pan += self.pan_velocity * dt
        self.tilt += self.tilt_velocity * dt

        # Enforce mechanical hard stops
        if self.pan < self.pan_limits[0]:
            self.pan = self.pan_limits[0]
            self.pan_velocity = max(0.0, self.pan_velocity)
        elif self.pan > self.pan_limits[1]:
            self.pan = self.pan_limits[1]
            self.pan_velocity = min(0.0, self.pan_velocity)

        if self.tilt < self.tilt_limits[0]:
            self.tilt = self.tilt_limits[0]
            self.tilt_velocity = max(0.0, self.tilt_velocity)
        elif self.tilt > self.tilt_limits[1]:
            self.tilt = self.tilt_limits[1]
            self.tilt_velocity = min(0.0, self.tilt_velocity)

    def project_target(
        self, target_az_rad: float, target_el_rad: float
    ) -> Tuple[float, float, float, float, bool]:
        """
        Project a target given its absolute world angular position.
        Returns:
            (delta_az, delta_el, pixel_u, pixel_v, is_in_fov)
        """
        delta_az = target_az_rad - self.effective_pan
        delta_el = target_el_rad - self.effective_tilt

        in_fov = self.projection.is_in_fov(delta_az, delta_el)
        u, v = self.projection.angles_to_pixels(delta_az, delta_el, exact=True)

        return delta_az, delta_el, u, v, in_fov

    def render_frame(
        self,
        beacons: List[Beacon],
        sim_time: float,
        base_canvas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Synthesize camera frame representing current field of view.
        """
        if base_canvas is not None:
            frame = base_canvas.copy()
        else:
            # Dark space/night background with minimal ambient photons
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Render each visible beacon into frame
        for beacon in beacons:
            if not beacon.is_emitting(sim_time):
                continue

            delta_az, delta_el, u, v, in_fov = self.project_target(
                beacon.pos[0], beacon.pos[1]
            )

            # Render if inside or within margin of sensor boundaries
            margin = 30
            if -margin <= u < self.width + margin and -margin <= v < self.height + margin:
                beacon.render_to_frame(frame, u, v)

        return frame
