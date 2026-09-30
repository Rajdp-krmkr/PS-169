"""
Trajectory models for simulating optical beacon motion in 2D angular space.

Supports:
- Linear (Constant Velocity)
- Circular / Orbital
- Sinusoidal (Wave / Flutter)
- Step / Random Maneuver
- Constant Acceleration
"""

from __future__ import annotations
import abc
import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class Trajectory(abc.ABC):
    """Abstract base class for target trajectories."""

    @abc.abstractmethod
    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        """
        Sample state at time t.
        Returns:
            (pos_x, pos_y, vel_x, vel_y, acc_x, acc_y) in radians (or degrees if configured).
        """
        pass


class LinearTrajectory(Trajectory):
    """Constant velocity motion with optional constant acceleration and boundary bouncing."""

    def __init__(
        self,
        x0: float,
        y0: float,
        vx: float,
        vy: float,
        ax: float = 0.0,
        ay: float = 0.0,
        bound_x: float = 24.0,
        bound_y: float = 14.0,
        bounce: bool = True,
    ) -> None:
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.vx = float(vx)
        self.vy = float(vy)
        self.ax = float(ax)
        self.ay = float(ay)
        self.bound_x = abs(float(bound_x))
        self.bound_y = abs(float(bound_y))
        self.bounce = bounce

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        raw_x = self.x0 + self.vx * t + 0.5 * self.ax * (t**2)
        raw_y = self.y0 + self.vy * t + 0.5 * self.ay * (t**2)
        raw_vx = self.vx + self.ax * t
        raw_vy = self.vy + self.ay * t

        if not self.bounce or self.bound_x <= 0.0:
            return raw_x, raw_y, raw_vx, raw_vy, self.ax, self.ay

        # Triangle fold reflection off boundaries [-bound_x, bound_x]
        span_x = 2.0 * self.bound_x
        folded_x = (raw_x - (-self.bound_x)) % (2.0 * span_x)
        if folded_x <= span_x:
            px = -self.bound_x + folded_x
            vx = raw_vx
            ax = self.ax
        else:
            px = self.bound_x - (folded_x - span_x)
            vx = -raw_vx
            ax = -self.ax

        span_y = 2.0 * self.bound_y
        folded_y = (raw_y - (-self.bound_y)) % (2.0 * span_y)
        if folded_y <= span_y:
            py = -self.bound_y + folded_y
            vy = raw_vy
            ay = self.ay
        else:
            py = self.bound_y - (folded_y - span_y)
            vy = -raw_vy
            ay = -self.ay

        return px, py, vx, vy, ax, ay


class CircularTrajectory(Trajectory):
    """Circular/orbital motion about a center point."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        angular_speed: float,
        initial_phase: float = 0.0,
    ) -> None:
        self.cx = center_x
        self.cy = center_y
        self.r = radius
        self.omega = angular_speed
        self.phase0 = initial_phase

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        theta = self.omega * t + self.phase0
        px = self.cx + self.r * math.cos(theta)
        py = self.cy + self.r * math.sin(theta)
        vx = -self.r * self.omega * math.sin(theta)
        vy = self.r * self.omega * math.cos(theta)
        ax = -self.r * (self.omega**2) * math.cos(theta)
        ay = -self.r * (self.omega**2) * math.sin(theta)
        return px, py, vx, vy, ax, ay


class SinusoidalTrajectory(Trajectory):
    """
    Sinusoidal trajectory with horizontal drift and vertical wave modulation.
    Features automatic smooth boundary reflection (harmonic turnaround) so the target
    remains bounded within the camera FOV and gimbal limits indefinitely.
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        vx: float,
        amplitude_y: float,
        frequency_hz: float,
        phase_y: float = 0.0,
        bound_x: float = 22.0,
        bounce: bool = True,
    ) -> None:
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.vx = float(vx)
        self.amp_y = float(amplitude_y)
        self.omega_y = 2.0 * math.pi * float(frequency_hz)
        self.phase_y = float(phase_y)
        self.bound_x = abs(float(bound_x))
        self.bounce = bounce

        if self.bounce and self.bound_x > 0.0:
            v_abs = max(abs(self.vx), 1e-4)
            # Harmonic frequency chosen so that peak speed at center equals |vx|
            self.omega_x = v_abs / self.bound_x
            clipped_ratio = max(-0.9999, min(0.9999, self.x0 / self.bound_x))
            self.phase_x = math.asin(clipped_ratio)
            if self.vx < 0:
                self.phase_x = math.pi - self.phase_x
        else:
            self.omega_x = 0.0
            self.phase_x = 0.0

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        theta_y = self.omega_y * t + self.phase_y
        py = self.y0 + self.amp_y * math.sin(theta_y)
        vy = self.amp_y * self.omega_y * math.cos(theta_y)
        ay = -self.amp_y * (self.omega_y**2) * math.sin(theta_y)

        if not self.bounce or self.bound_x <= 0.0:
            px = self.x0 + self.vx * t
            vx = self.vx
            ax = 0.0
        else:
            theta_x = self.omega_x * t + self.phase_x
            px = self.bound_x * math.sin(theta_x)
            vx = self.bound_x * self.omega_x * math.cos(theta_x)
            ax = -self.bound_x * (self.omega_x**2) * math.sin(theta_x)

        return px, py, vx, vy, ax, ay


class ManeuverTrajectory(Trajectory):
    """
    Simulates unpredictable target maneuvers (e.g. UAV buffeting, avoidance maneuvers)
    with smooth cubic spline or periodic random acceleration changes.
    Dynamically extends maneuver segments indefinitely and keeps motion bounded.
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        vx0: float,
        vy0: float,
        max_acc: float = 1.0,
        maneuver_interval: float = 2.0,
        seed: int = 42,
        bound_x: float = 22.0,
        bound_y: float = 13.0,
    ) -> None:
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.vx0 = float(vx0)
        self.vy0 = float(vy0)
        self.max_acc = float(max_acc)
        self.interval = float(maneuver_interval)
        self.rng = np.random.default_rng(seed)
        self.bound_x = abs(float(bound_x))
        self.bound_y = abs(float(bound_y))

        # Precompute initial maneuver segments
        self.segments: List[Tuple[float, float, float, float, float, float, float, float]] = []
        self._ensure_segments(120.0)

    def _ensure_segments(self, target_t: float) -> None:
        """Dynamically generate maneuver segments up to target_t + safety horizon."""
        horizon_t = target_t + 60.0
        while not self.segments or self.segments[-1][1] < horizon_t:
            if not self.segments:
                cur_t = 0.0
                cur_x = self.x0
                cur_y = self.y0
                cur_vx = self.vx0
                cur_vy = self.vy0
            else:
                last = self.segments[-1]
                cur_t = last[1]
                dur_last = last[1] - last[0]
                cur_x = last[2] + last[4] * dur_last + 0.5 * last[6] * (dur_last**2)
                cur_y = last[3] + last[5] * dur_last + 0.5 * last[7] * (dur_last**2)
                cur_vx = last[4] + last[6] * dur_last
                cur_vy = last[5] + last[7] * dur_last

            dur = float(self.rng.uniform(self.interval * 0.5, self.interval * 1.5))

            # Smooth proportional restoring acceleration toward center
            restoring_ax = -(cur_x / max(self.bound_x, 1e-3)) * self.max_acc
            restoring_ay = -(cur_y / max(self.bound_y, 1e-3)) * self.max_acc

            ax = float(self.rng.uniform(-self.max_acc * 0.5, self.max_acc * 0.5)) + restoring_ax
            ay = float(self.rng.uniform(-self.max_acc * 0.5, self.max_acc * 0.5)) + restoring_ay

            # Turn around / damp velocity if approaching boundary
            if abs(cur_x) > self.bound_x * 0.75 and cur_x * cur_vx > 0:
                cur_vx *= -0.5
            if abs(cur_y) > self.bound_y * 0.75 and cur_y * cur_vy > 0:
                cur_vy *= -0.5

            max_speed = 3.5
            cur_vx = float(np.clip(cur_vx, -max_speed, max_speed))
            cur_vy = float(np.clip(cur_vy, -max_speed, max_speed))

            self.segments.append((cur_t, cur_t + dur, cur_x, cur_y, cur_vx, cur_vy, ax, ay))

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        self._ensure_segments(t)
        for t_start, t_end, x_s, y_s, vx_s, vy_s, ax, ay in self.segments:
            if t_start <= t < t_end:
                dt = t - t_start
                px = x_s + vx_s * dt + 0.5 * ax * (dt**2)
                py = y_s + vy_s * dt + 0.5 * ay * (dt**2)
                vx = vx_s + ax * dt
                vy = vy_s + ay * dt
                return px, py, vx, vy, ax, ay

        # Fallback to last segment extrapolation if precisely on boundary
        last = self.segments[-1]
        dt = t - last[0]
        return (
            last[2] + last[4] * dt,
            last[3] + last[5] * dt,
            last[4],
            last[5],
            0.0,
            0.0,
        )


def create_trajectory(cfg: Dict[str, Any]) -> Trajectory:
    """Factory helper to build a trajectory instance from config dictionary."""
    traj_type = cfg.get("type", "linear").lower()

    if traj_type == "linear":
        return LinearTrajectory(
            x0=cfg.get("x0", 0.0),
            y0=cfg.get("y0", 0.0),
            vx=cfg.get("vx", cfg.get("vx_deg_s", 0.0)),
            vy=cfg.get("vy", cfg.get("vy_deg_s", 0.0)),
            ax=cfg.get("ax", 0.0),
            ay=cfg.get("ay", 0.0),
            bound_x=cfg.get("bound_x_deg", 24.0),
            bound_y=cfg.get("bound_y_deg", 14.0),
            bounce=cfg.get("bounce", True),
        )
    elif traj_type == "circular":
        center = cfg.get("center_deg", [0.0, 0.0])
        return CircularTrajectory(
            center_x=center[0],
            center_y=center[1],
            radius=cfg.get("radius_deg", 5.0),
            angular_speed=cfg.get("angular_speed_deg_s", 10.0),
            initial_phase=cfg.get("initial_phase_rad", 0.0),
        )
    elif traj_type == "sinusoidal":
        center = cfg.get("center_deg", [0.0, 0.0])
        return SinusoidalTrajectory(
            x0=center[0],
            y0=center[1],
            vx=cfg.get("vx_deg_s", 1.0),
            amplitude_y=cfg.get("vy_amplitude_deg", 2.0),
            frequency_hz=cfg.get("vy_frequency_hz", 0.2),
            phase_y=cfg.get("phase_rad", 0.0),
            bound_x=cfg.get("bound_x_deg", 22.0),
            bounce=cfg.get("bounce", True),
        )
    elif traj_type == "maneuver":
        return ManeuverTrajectory(
            x0=cfg.get("x0", 0.0),
            y0=cfg.get("y0", 0.0),
            vx0=cfg.get("vx_deg_s", 1.0),
            vy0=cfg.get("vy_deg_s", 0.0),
            max_acc=cfg.get("max_acc", 1.5),
            maneuver_interval=cfg.get("maneuver_interval_s", 2.0),
            seed=cfg.get("seed", 42),
            bound_x=cfg.get("bound_x_deg", 22.0),
            bound_y=cfg.get("bound_y_deg", 13.0),
        )
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
