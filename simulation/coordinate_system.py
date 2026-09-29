"""
Coordinate system transformations and optical projections for FSOC Coarse PAT.

Conventions:
- World angles: theta_x (Azimuth/Horizontal), theta_y (Elevation/Vertical).
  Units: Radians internally, degrees in GUI and configuration.
- Camera pointing: theta_pan (Azimuth), theta_tilt (Elevation).
- Relative angle error:
    delta_theta_x = theta_target_x - theta_pan
    delta_theta_y = theta_target_y - theta_tilt
- Image plane:
    u: Horizontal pixel index, 0 to (width - 1), increases left to right.
    v: Vertical pixel index, 0 to (height - 1), increases top to bottom.
    c_x = width / 2.0, c_y = height / 2.0 (Optical center).
    Positive elevation (tilt) maps to upper image region (smaller v).
"""

from __future__ import annotations
import math
from typing import Tuple


def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * (math.pi / 180.0)


def rad2deg(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * (180.0 / math.pi)


def normalize_angle_rad(rad: float) -> float:
    """Normalize angle to (-pi, pi] radians."""
    res = (rad + math.pi) % (2.0 * math.pi) - math.pi
    if res <= -math.pi:
        res += 2.0 * math.pi
    return res


def normalize_angle_deg(deg: float) -> float:
    """Normalize angle to (-180, 180] degrees."""
    res = (deg + 180.0) % 360.0 - 180.0
    if res <= -180.0:
        res += 360.0
    return res


class OpticalProjection:
    """
    Pinhole optical projection model with exact trigonometric and small-angle options.
    """

    def __init__(
        self,
        resolution: Tuple[int, int],
        fov_horizontal_rad: float,
        fov_vertical_rad: float,
    ) -> None:
        self.width, self.height = resolution
        self.fov_h = fov_horizontal_rad
        self.fov_v = fov_vertical_rad

        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # Exact pinhole focal lengths in pixel units
        self.fx = (self.width / 2.0) / math.tan(self.fov_h / 2.0)
        self.fy = (self.height / 2.0) / math.tan(self.fov_v / 2.0)

        # Linear approximation scale factors (rad per pixel)
        self.rad_per_pixel_x = self.fov_h / self.width
        self.rad_per_pixel_y = self.fov_v / self.height

    def is_in_fov(self, delta_az_rad: float, delta_el_rad: float) -> bool:
        """Check if relative angular target coordinates fall inside camera FOV."""
        return (abs(delta_az_rad) <= self.fov_h / 2.0) and (
            abs(delta_el_rad) <= self.fov_v / 2.0
        )

    def angles_to_pixels(
        self, delta_az_rad: float, delta_el_rad: float, exact: bool = True
    ) -> Tuple[float, float]:
        """
        Project relative angular offsets (rad) onto camera image plane (u, v in pixels).
        u: 0 to width
        v: 0 to height (elevation increases upward -> lower pixel v)
        """
        if exact:
            u = self.cx + self.fx * math.tan(delta_az_rad)
            v = self.cy - self.fy * math.tan(delta_el_rad)
        else:
            u = self.cx + (delta_az_rad / self.fov_h) * self.width
            v = self.cy - (delta_el_rad / self.fov_v) * self.height
        return u, v

    def pixels_to_angular_error(
        self, u: float, v: float, exact: bool = True
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinate (u, v) on image sensor to angular error (delta_az, delta_el) in radians.
        Positive delta_az: target is to the right of boresight -> camera needs to pan right (positive).
        Positive delta_el: target is above boresight -> camera needs to tilt up (positive).
        """
        du = u - self.cx
        dv = self.cy - v  # Note: target above center has v < cy, so dv > 0

        if exact:
            delta_az = math.atan(du / self.fx)
            delta_el = math.atan(dv / self.fy)
        else:
            delta_az = du * self.rad_per_pixel_x
            delta_el = dv * self.rad_per_pixel_y

        return delta_az, delta_el

    def pixel_distance_to_center(self, u: float, v: float) -> float:
        """Compute Euclidean distance in pixels from optical center."""
        return math.hypot(u - self.cx, v - self.cy)

    def angular_distance(self, delta_az_rad: float, delta_el_rad: float) -> float:
        """Compute Euclidean angular distance in radians."""
        return math.hypot(delta_az_rad, delta_el_rad)
