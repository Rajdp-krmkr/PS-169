"""
Image preprocessing utilities for FSOC beacon detection.
"""

from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    gaussian_kernel: int = 3,
    use_clahe: bool = False,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess image frame:
    - Grayscale conversion
    - Optional ROI cropping
    - Gaussian smoothing to suppress high-frequency noise
    - Optional Contrast Limited Adaptive Histogram Equalization (CLAHE)
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    if roi is not None:
        rx, ry, rw, rh = roi
        h, w = gray.shape[:2]
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(1, min(rw, w - rx))
        rh = max(1, min(rh, h - ry))
        gray = gray[ry : ry + rh, rx : rx + rw]

    if gaussian_kernel > 1:
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray, None
