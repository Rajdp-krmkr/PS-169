"""
Optical blur models: Directional Motion Blur and Optical Defocus Blur.
"""

from __future__ import annotations
import math
import numpy as np
import cv2


class MotionBlur:
    """
    Directional motion blur caused by rapid gimbal slew or target motion during exposure.
    """

    def __init__(self, max_kernel_size: int = 15) -> None:
        self.max_ksize = max_kernel_size

    def apply(
        self,
        frame: np.ndarray,
        pan_rate_rad_s: float,
        tilt_rate_rad_s: float,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        Apply directional motion blur kernel aligned with camera motion vector.
        """
        if strength <= 0.0:
            return frame

        speed = math.hypot(pan_rate_rad_s, tilt_rate_rad_s)
        ksize = int(round(min(self.max_ksize, speed * 25.0 * strength)))
        if ksize < 3:
            return frame

        if ksize % 2 == 0:
            ksize += 1

        angle_deg = math.degrees(math.atan2(-tilt_rate_rad_s, pan_rate_rad_s))

        # Create 1D linear blur kernel and rotate it
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        mid = ksize // 2
        kernel[mid, :] = 1.0

        M = cv2.getRotationMatrix2D((mid, mid), angle_deg, 1.0)
        rotated_kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        k_sum = np.sum(rotated_kernel)
        if k_sum > 0:
            rotated_kernel /= k_sum

        return cv2.filter2D(frame, -1, rotated_kernel)


class DefocusBlur:
    """
    Optical defocus blur (Airy disk expansion / Gaussian defocus).
    """

    def __init__(self, max_sigma: float = 4.0) -> None:
        self.max_sigma = max_sigma

    def apply(self, frame: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if strength <= 0.0:
            return frame

        sigma = self.max_sigma * strength
        ksize = int(math.ceil(sigma * 3.0)) * 2 + 1
        if ksize < 3:
            return frame

        return cv2.GaussianBlur(frame, (ksize, ksize), sigma)
