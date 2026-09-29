"""
Unit tests for AI YOLO detector loading and fallback behavior.
"""

import numpy as np
import pytest
from vision.yolo_detector import YOLODetector


def test_yolo_fallback_when_model_missing():
    # Pass non-existent path
    detector = YOLODetector(model_path="models/non_existent_weight.onnx")
    assert detector.is_loaded is False

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(frame)
    assert res.success is False
    assert res.primary is None
    assert len(res.candidates) == 0
