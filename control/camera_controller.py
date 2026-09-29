"""
Closed-loop 2-axis Pan-Tilt Camera Gimbal Controller for FSOC coarse tracking.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from simulation.coordinate_system import deg2rad, rad2deg
from simulation.camera import VirtualCamera
from control.pid import PIDController
from control.latency import ControlDelayQueue


@dataclass
class ControllerTelemetry:
    delta_az_rad: float = 0.0
    delta_el_rad: float = 0.0
    delta_az_deg: float = 0.0
    delta_el_deg: float = 0.0
    cmd_pan_rate_rad_s: float = 0.0
    cmd_tilt_rate_rad_s: float = 0.0
    pan_p: float = 0.0
    pan_i: float = 0.0
    pan_d: float = 0.0
    tilt_p: float = 0.0
    tilt_i: float = 0.0
    tilt_d: float = 0.0
    is_in_deadzone: bool = False


class CameraGimbalController:
    """
    Coordinates dual-axis PID tracking of an optical beacon with dead-zone locking.
    """

    def __init__(
        self,
        camera: VirtualCamera,
        kp_pan: float = 1.0,
        ki_pan: float = 0.02,
        kd_pan: float = 0.15,
        kp_tilt: float = 1.0,
        ki_tilt: float = 0.02,
        kd_tilt: float = 0.15,
        dead_zone_deg: float = 0.04,
        integral_limit_deg: float = 2.0,
        derivative_filter_alpha: float = 0.8,
        latency_frames: int = 0,
    ) -> None:
        self.camera = camera
        self.dead_zone_rad = deg2rad(dead_zone_deg)
        int_limit_rad = deg2rad(integral_limit_deg)
        self.delay_queue = ControlDelayQueue(delay_frames=latency_frames)

        self.pid_pan = PIDController(
            kp=kp_pan,
            ki=ki_pan,
            kd=kd_pan,
            dead_zone=self.dead_zone_rad,
            output_limits=(-camera.max_pan_speed, camera.max_pan_speed),
            integral_limits=(-int_limit_rad, int_limit_rad),
            derivative_filter_alpha=derivative_filter_alpha,
        )

        self.pid_tilt = PIDController(
            kp=kp_tilt,
            ki=ki_tilt,
            kd=kd_tilt,
            dead_zone=self.dead_zone_rad,
            output_limits=(-camera.max_tilt_speed, camera.max_tilt_speed),
            integral_limits=(-int_limit_rad, int_limit_rad),
            derivative_filter_alpha=derivative_filter_alpha,
        )

        self.telemetry = ControllerTelemetry()

    def set_gains(
        self,
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None,
        dead_zone_deg: Optional[float] = None,
    ) -> None:
        """Update PID parameters in real-time."""
        if kp is not None:
            self.pid_pan.kp = kp
            self.pid_tilt.kp = kp
        if ki is not None:
            self.pid_pan.ki = ki
            self.pid_tilt.ki = ki
        if kd is not None:
            self.pid_pan.kd = kd
            self.pid_tilt.kd = kd
        if dead_zone_deg is not None:
            self.dead_zone_rad = deg2rad(dead_zone_deg)
            self.pid_pan.dead_zone = self.dead_zone_rad
            self.pid_tilt.dead_zone = self.dead_zone_rad

    def reset(self) -> None:
        """Reset controllers."""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        self.delay_queue.reset()
        self.camera.set_command_rate(0.0, 0.0)

    def track_pixel_error(
        self, pixel_u: float, pixel_v: float, dt: float
    ) -> ControllerTelemetry:
        """
        Closed-loop track target from image sensor coordinates (u, v).
        """
        # Convert pixel position to optical boresight error angles
        delta_az, delta_el = self.camera.projection.pixels_to_angular_error(
            pixel_u, pixel_v, exact=True
        )

        return self.track_angular_error(delta_az, delta_el, dt)

    def track_angular_error(
        self, delta_az_rad: float, delta_el_rad: float, dt: float
    ) -> ControllerTelemetry:
        """
        Closed-loop track target from angular errors (delta_az, delta_el in rad).
        """
        # Desired camera movement: positive delta_az -> rotate pan positive
        raw_cmd_pan = self.pid_pan.compute(delta_az_rad, dt)
        raw_cmd_tilt = self.pid_tilt.compute(delta_el_rad, dt)

        # Buffer through transport delay queue
        cmd_pan_rate, cmd_tilt_rate = self.delay_queue.step(raw_cmd_pan, raw_cmd_tilt)

        # Command virtual gimbal
        self.camera.set_command_rate(cmd_pan_rate, cmd_tilt_rate)

        in_deadzone = (abs(delta_az_rad) <= self.dead_zone_rad) and (
            abs(delta_el_rad) <= self.dead_zone_rad
        )

        self.telemetry = ControllerTelemetry(
            delta_az_rad=delta_az_rad,
            delta_el_rad=delta_el_rad,
            delta_az_deg=rad2deg(delta_az_rad),
            delta_el_deg=rad2deg(delta_el_rad),
            cmd_pan_rate_rad_s=cmd_pan_rate,
            cmd_tilt_rate_rad_s=cmd_tilt_rate,
            pan_p=self.pid_pan.p_term,
            pan_i=self.pid_pan.i_term,
            pan_d=self.pid_pan.d_term,
            tilt_p=self.pid_tilt.p_term,
            tilt_i=self.pid_tilt.i_term,
            tilt_d=self.pid_tilt.d_term,
            is_in_deadzone=in_deadzone,
        )

        return self.telemetry

    def manual_jog(self, pan_rate_deg_s: float, tilt_rate_deg_s: float) -> None:
        """Manual joystick/jog command override."""
        self.camera.set_command_rate(
            deg2rad(pan_rate_deg_s), deg2rad(tilt_rate_deg_s)
        )
