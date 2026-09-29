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
from typing import Dict, Any, Tuple
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
    """Constant velocity motion with optional constant acceleration."""

    def __init__(
        self,
        x0: float,
        y0: float,
        vx: float,
        vy: float,
        ax: float = 0.0,
        ay: float = 0.0,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        px = self.x0 + self.vx * t + 0.5 * self.ax * (t**2)
        py = self.y0 + self.vy * t + 0.5 * self.ay * (t**2)
        vx = self.vx + self.ax * t
        vy = self.vy + self.ay * t
        return px, py, vx, vy, self.ax, self.ay


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
    """Linear horizontal drift with sinusoidal vertical modulation."""

    def __init__(
        self,
        x0: float,
        y0: float,
        vx: float,
        amplitude_y: float,
        frequency_hz: float,
        phase_y: float = 0.0,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.amp_y = amplitude_y
        self.omega_y = 2.0 * math.pi * frequency_hz
        self.phase_y = phase_y

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        theta = self.omega_y * t + self.phase_y
        px = self.x0 + self.vx * t
        py = self.y0 + self.amp_y * math.sin(theta)
        vx = self.vx
        vy = self.amp_y * self.omega_y * math.cos(theta)
        ax = 0.0
        ay = -self.amp_y * (self.omega_y**2) * math.sin(theta)
        return px, py, vx, vy, ax, ay


class ManeuverTrajectory(Trajectory):
    """
    Simulates unpredictable target maneuvers (e.g. UAV buffeting, avoidance maneuvers)
    with smooth cubic spline or periodic random acceleration changes.
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
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.vx0 = vx0
        self.vy0 = vy0
        self.max_acc = max_acc
        self.interval = maneuver_interval
        self.rng = np.random.default_rng(seed)

        # Precompute maneuver segments
        self.segments = []
        cur_t = 0.0
        cur_x = x0
        cur_y = y0
        cur_vx = vx0
        cur_vy = vy0

        for _ in range(50):
            ax = float(self.rng.uniform(-max_acc, max_acc))
            ay = float(self.rng.uniform(-max_acc, max_acc))
            dur = float(self.rng.uniform(self.interval * 0.5, self.interval * 1.5))
            self.segments.append((cur_t, cur_t + dur, cur_x, cur_y, cur_vx, cur_vy, ax, ay))
            cur_x += cur_vx * dur + 0.5 * ax * (dur**2)
            cur_y += cur_vy * dur + 0.5 * ay * (dur**2)
            cur_vx += ax * dur
            cur_vy += ay * dur
            cur_t += dur

    def sample(self, t: float) -> Tuple[float, float, float, float, float, float]:
        for t_start, t_end, x_s, y_s, vx_s, vy_s, ax, ay in self.segments:
            if t_start <= t < t_end:
                dt = t - t_start
                px = x_s + vx_s * dt + 0.5 * ax * (dt**2)
                py = y_s + vy_s * dt + 0.5 * ay * (dt**2)
                vx = vx_s + ax * dt
                vy = vy_s + ay * dt
                return px, py, vx, vy, ax, ay

        # Fallback to last segment extrapolation
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
        )
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
