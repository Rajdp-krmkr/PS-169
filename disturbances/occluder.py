"""
Dynamic Line-of-Sight Occluder Simulation for FSOC PAT.

Simulates physical opaque or translucent obstacles (e.g. atmospheric cloud banks,
aerosol plumes, birds, space debris) drifting across the optical line-of-sight.

Evaluates:
- Target tracking coasting under temporary LOS blackout.
- False alarm rejection (ensuring tracker doesn't latch onto cloud edges).
- Automatic reacquisition latency when target clears obstacle.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np

from simulation.coordinate_system import rad2deg, deg2rad
from simulation.camera import VirtualCamera


class DynamicOccluder:
    """
    Physical moving obstacle in angular space that can block the optical beacon.
    """

    def __init__(
        self,
        initial_pos_rad: Tuple[float, float] = (-0.02, 0.0),
        velocity_rad_s: Tuple[float, float] = (0.015, 0.0),
        radius_rad: float = 0.012,  # ~0.69 deg radius
        opacity: float = 0.95,
    ) -> None:
        self.az_rad = float(initial_pos_rad[0])
        self.el_rad = float(initial_pos_rad[1])
        self.vx_rad_s = float(velocity_rad_s[0])
        self.vy_rad_s = float(velocity_rad_s[1])
        self.radius_rad = float(radius_rad)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))

    def step(self, dt: float) -> None:
        """Advance occluder kinematic position."""
        self.az_rad += self.vx_rad_s * dt
        self.el_rad += self.vy_rad_s * dt

    def is_occluding(self, target_az_rad: float, target_el_rad: float) -> bool:
        """Check if target point falls within occluder radius."""
        dist = math.hypot(target_az_rad - self.az_rad, target_el_rad - self.el_rad)
        return dist <= self.radius_rad

    def get_transmission_factor(self, target_az_rad: float, target_el_rad: float) -> float:
        """
        Return optical transmission factor [0.0 to 1.0] through the occluder.
        1.0 = clear unobstructed transmission, 0.0 = completely blocked.
        """
        dist = math.hypot(target_az_rad - self.az_rad, target_el_rad - self.el_rad)
        if dist > self.radius_rad:
            return 1.0
        # Soft Gaussian/cos falloff toward edges
        norm_dist = dist / max(1e-6, self.radius_rad)
        attenuation = self.opacity * (1.0 - norm_dist**2)
        return max(0.0, min(1.0, 1.0 - attenuation))

    def render_overlay(self, camera: VirtualCamera, canvas: np.ndarray) -> np.ndarray:
        """
        Render semi-transparent cloud obstacle on camera viewport frame.
        """
        _, _, u, v, in_fov = camera.project_target(self.az_rad, self.el_rad)
        # Compute radius in pixels
        w, _ = camera.resolution
        fov_h = getattr(camera, "fov_h_rad", camera.fov_horizontal_rad)
        r_pix = int(round((self.radius_rad / fov_h) * w))
        if r_pix < 2:
            return canvas

        h_canvas, w_canvas = canvas.shape[:2]
        if not (-r_pix <= u < w_canvas + r_pix and -r_pix <= v < h_canvas + r_pix):
            return canvas

        overlay = canvas.copy()
        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            r_pix,
            (45, 48, 55),
            -1,
            lineType=cv2.LINE_AA,
        )
        # Cloud border ring
        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            r_pix,
            (70, 75, 85),
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, self.opacity * 0.75, canvas, 1.0 - (self.opacity * 0.75), 0, canvas)
        return canvas


class OccluderManager:
    """
    Manages multiple dynamic scheduled occluders.
    """

    def __init__(self) -> None:
        self.occluders: List[DynamicOccluder] = []

    def add_occluder(self, occluder: DynamicOccluder) -> None:
        self.occluders.append(occluder)

    def clear(self) -> None:
        self.occluders.clear()

    def step(self, dt: float) -> None:
        for occ in self.occluders:
            occ.step(dt)

    def check_occlusion(self, target_az_rad: float, target_el_rad: float) -> Tuple[bool, float]:
        """
        Returns: (is_blocked_bool, total_transmission_factor)
        """
        transmission = 1.0
        is_blocked = False
        for occ in self.occluders:
            if occ.is_occluding(target_az_rad, target_el_rad):
                is_blocked = True
            transmission *= occ.get_transmission_factor(target_az_rad, target_el_rad)
        return is_blocked, transmission

    def render_all(self, camera: VirtualCamera, canvas: np.ndarray) -> np.ndarray:
        out = canvas
        for occ in self.occluders:
            out = occ.render_overlay(camera, out)
        return out
