"""
Sensor noise disturbance models for optical detectors (Gaussian and Salt-and-Pepper).
"""

from __future__ import annotations
import numpy as np
import cv2


class SensorNoise:
    """
    Simulates optical detector shot noise, dark current, and pixel defects.
    """

    def __init__(
        self,
        gaussian_std: float = 0.0,
        salt_pepper_prob: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.gaussian_std = gaussian_std
        self.sp_prob = salt_pepper_prob
        self.rng = np.random.default_rng(seed)

    def apply(self, frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply noise to image frame."""
        if strength <= 0.0:
            return frame

        noisy = frame.astype(np.float32)

        # 1. Gaussian sensor noise
        effective_std = self.gaussian_std * strength
        if effective_std > 0.0:
            noise = self.rng.normal(0.0, effective_std, frame.shape)
            noisy += noise

        # 2. Salt and pepper noise
        effective_sp = self.sp_prob * strength
        if effective_sp > 0.0:
            num_sp = int(frame.shape[0] * frame.shape[1] * effective_sp)
            if num_sp > 0:
                coords_y = self.rng.integers(0, frame.shape[0], num_sp)
                coords_x = self.rng.integers(0, frame.shape[1], num_sp)
                vals = self.rng.choice([0, 255], size=num_sp).astype(np.float32)
                for c in range(noisy.shape[2] if noisy.ndim == 3 else 1):
                    if noisy.ndim == 3:
                        noisy[coords_y, coords_x, c] = vals
                    else:
                        noisy[coords_y, coords_x] = vals

        return np.clip(noisy, 0, 255).astype(np.uint8)
