"""
Unit tests for classical computer vision optical beacon detector.
"""

import math
import numpy as np
import pytest
from vision.classical_detector import ClassicalBeaconDetector
from simulation.beacon import Beacon


def test_classical_detector_clean_spot():
    detector = ClassicalBeaconDetector(binary_threshold=140, min_area=4.0)

    # Synthetic black frame with a single bright Gaussian spot at (320, 240)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    beacon = Beacon(
        beacon_id=1,
        initial_pos_rad=(0.0, 0.0),
        intensity=255.0,
        radius_pixels=8.0,
        color_bgr=(255, 255, 255),
    )
    beacon.render_to_frame(frame, pixel_u=320.0, pixel_v=240.0)

    result = detector.detect(frame, timestamp=0.0)
    assert result.success is True
    assert result.primary is not None
    assert math.isclose(result.primary.center_u, 320.0, abs_tol=1.5)
    assert math.isclose(result.primary.center_v, 240.0, abs_tol=1.5)
    assert result.primary.confidence > 0.7


def test_classical_detector_distractor_filtering():
    detector = ClassicalBeaconDetector(
        binary_threshold=140,
        min_circularity=0.6,
        min_peak_intensity=150.0,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Render a low-brightness noise source
    dim_beacon = Beacon(
        beacon_id=2,
        intensity=90.0,  # Below min_peak_intensity (150)
        radius_pixels=6.0,
    )
    dim_beacon.render_to_frame(frame, pixel_u=100.0, pixel_v=100.0)

    result = detector.detect(frame, timestamp=0.0)
    assert result.success is False
    assert result.primary is None
