"""
Visual atmospheric turbulence simulation model for FSOC optical tracking.

Note:
This is a computer-vision simulation approximation designed for testing
detection and tracking robustness under atmospheric scintillation and beam wander,
rather than a full Kolmogorov phase screen physical wave propagation solver.

Implementation:
Generates a low-frequency 2D spatial displacement vector field deformed temporally,
smoothed with spatial Gaussian filtering, and applied to the frame via cv2.remap.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np
import cv2


class AtmosphericTurbulence:
    """
    Simulates optical beam wander, scintillation, and spatial distortion.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        max_displacement_pixels: float = 8.0,
        temporal_frequency_hz: float = 1.5,
        grid_step: int = 32,
        seed: int = 42,
    ) -> None:
        self.width, self.height = resolution
        self.max_disp = max_displacement_pixels
        self.temp_freq = temporal_frequency_hz
        self.grid_step = grid_step
        self.rng = np.random.default_rng(seed)

        # Base identity coordinate meshgrid
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(self.width, dtype=np.float32),
            np.arange(self.height, dtype=np.float32),
        )

        # Pre-generate low-resolution random spatial harmonic vectors
        gw = self.width // self.grid_step + 2
        gh = self.height // self.grid_step + 2
        self.base_dx1 = self.rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
        self.base_dy1 = self.rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
        self.base_dx2 = self.rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
        self.base_dy2 = self.rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)

    def apply(
        self, frame: np.ndarray, sim_time: float, strength: float = 1.0
    ) -> np.ndarray:
        """
        Warp frame using time-varying spatial displacement field.
        """
        if strength <= 0.0:
            return frame

        h, w = frame.shape[:2]
        # Reinitialize meshgrid if resolution changed
        if w != self.width or h != self.height:
            self.width, self.height = w, h
            self.grid_x, self.grid_y = np.meshgrid(
                np.arange(self.width, dtype=np.float32),
                np.arange(self.height, dtype=np.float32),
            )

        omega = 2.0 * math.pi * self.temp_freq
        phase1 = math.sin(omega * sim_time)
        phase2 = math.cos(omega * 0.7 * sim_time + 1.0)

        # Low-frequency displacement field
        low_dx = (self.base_dx1 * phase1 + self.base_dx2 * phase2) * (
            self.max_disp * strength
        )
        low_dy = (self.base_dy1 * phase2 + self.base_dy2 * phase1) * (
            self.max_disp * strength
        )

        # Upscale and smoothly interpolate displacement to full resolution
        dx_map = cv2.resize(low_dx, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        dy_map = cv2.resize(low_dy, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        map_x = self.grid_x + dx_map
        map_y = self.grid_y + dy_map

        warped = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return warped
