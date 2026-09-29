"""
Unit tests for multi-modal hybrid detection fusion.
"""

import math
import pytest
from vision.detection_types import Detection, BoundingBox, DetectionResult
from vision.fusion import DetectionFusion


def test_fusion_matching():
    fusion = DetectionFusion(
        weight_ai=0.40,
        weight_brightness=0.30,
        weight_shape=0.15,
        weight_temporal=0.15,
        association_radius_pixels=25.0,
    )

    # Classical detection at (300, 200)
    det_cv = Detection(
        center_u=300.0,
        center_v=200.0,
        bbox=BoundingBox(x=290, y=190, w=20, h=20),
        confidence=0.85,
        area=120.0,
        peak_intensity=255.0,
        circularity=0.90,
        detector_name="classical_cv",
        timestamp=0.0,
    )
    res_cv = DetectionResult(primary=det_cv, candidates=[det_cv], success=True)

    # AI detection at (302, 201) (overlapping candidate)
    det_ai = Detection(
        center_u=302.0,
        center_v=201.0,
        bbox=BoundingBox(x=291, y=191, w=22, h=22),
        confidence=0.92,
        area=130.0,
        peak_intensity=255.0,
        circularity=0.95,
        detector_name="ai_yolo",
        timestamp=0.0,
    )
    res_ai = DetectionResult(primary=det_ai, candidates=[det_ai], success=True)

    fused_res = fusion.fuse(res_cv, res_ai, timestamp=0.0)
    assert fused_res.success is True
    assert fused_res.primary is not None

    # Fused center should be between (300, 200) and (302, 201)
    assert 300.0 <= fused_res.primary.center_u <= 302.0
    assert 200.0 <= fused_res.primary.center_v <= 201.0
    assert fused_res.primary.detector_name == "hybrid_fusion_matched"
    assert fused_res.primary.confidence > 0.80


def test_fusion_fallback_without_ai():
    fusion = DetectionFusion()
    det_cv = Detection(
        center_u=150.0,
        center_v=150.0,
        bbox=BoundingBox(x=140, y=140, w=20, h=20),
        confidence=0.80,
        area=100.0,
        peak_intensity=240.0,
        circularity=0.85,
        detector_name="classical_cv",
        timestamp=0.0,
    )
    res_cv = DetectionResult(primary=det_cv, candidates=[det_cv], success=True)
    res_ai_empty = DetectionResult(primary=None, candidates=[], success=False)

    fused_res = fusion.fuse(res_cv, res_ai_empty, timestamp=0.0)
    assert fused_res.success is True
    assert fused_res.primary is not None
    assert fused_res.primary.detector_name == "hybrid_cv_only"
    assert math.isclose(fused_res.primary.center_u, 150.0, abs_tol=1e-5)
