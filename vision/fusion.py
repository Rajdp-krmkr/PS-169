"""
Multi-Modal Hybrid Detection Fusion for FSOC Optical PAT.

Fuses:
1. Classical Computer Vision (Shape, Contours, Point-Spread Profile)
2. AI Object Detection (Deep Feature Pattern Recognition)
3. Temporal Track Continuity (Temporal Proximity Scoring)

Formula:
C_total = w_ai * C_ai + w_bright * C_bright + w_shape * C_shape + w_temp * C_temp
"""

from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from vision.detection_types import Detection, BoundingBox, DetectionResult


class DetectionFusion:
    """
    Associates and fuses classical CV and AI detections into high-confidence unified tracks.
    """

    def __init__(
        self,
        weight_ai: float = 0.35,
        weight_brightness: float = 0.25,
        weight_shape: float = 0.20,
        weight_temporal: float = 0.20,
        association_radius_pixels: float = 30.0,
        temporal_sigma_pixels: float = 120.0,
    ) -> None:
        self.w_ai = weight_ai
        self.w_bright = weight_brightness
        self.w_shape = weight_shape
        self.w_temp = weight_temporal
        self.assoc_radius = association_radius_pixels
        self.temp_sigma = temporal_sigma_pixels

        self.last_track_pos: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        """Reset temporal state."""
        self.last_track_pos = None

    def fuse(
        self,
        classical_result: DetectionResult,
        ai_result: DetectionResult,
        timestamp: float = 0.0,
    ) -> DetectionResult:
        """
        Merge candidate detections from both detectors.
        """
        start_t = time.perf_counter()

        cv_candidates = classical_result.candidates
        ai_candidates = ai_result.candidates

        matched_ai_indices = set()
        fused_candidates: List[Detection] = []

        # 1. Match Classical candidates with AI candidates
        for cv_det in cv_candidates:
            best_ai_idx = -1
            best_dist = float("inf")

            for idx, ai_det in enumerate(ai_candidates):
                if idx in matched_ai_indices:
                    continue
                dist = math.hypot(
                    cv_det.center_u - ai_det.center_u,
                    cv_det.center_v - ai_det.center_v,
                )
                if dist < self.assoc_radius and dist < best_dist:
                    best_dist = dist
                    best_ai_idx = idx

            if best_ai_idx >= 0:
                # Fused match!
                ai_match = ai_candidates[best_ai_idx]
                matched_ai_indices.add(best_ai_idx)

                # Weighted center based on confidence
                w_total = cv_det.confidence + ai_match.confidence
                u_fused = (
                    cv_det.center_u * cv_det.confidence + ai_match.center_u * ai_match.confidence
                ) / max(1e-5, w_total)
                v_fused = (
                    cv_det.center_v * cv_det.confidence + ai_match.center_v * ai_match.confidence
                ) / max(1e-5, w_total)

                # Union bounding box
                bx1 = min(cv_det.bbox.x, ai_match.bbox.x)
                by1 = min(cv_det.bbox.y, ai_match.bbox.y)
                bx2 = max(cv_det.bbox.x + cv_det.bbox.w, ai_match.bbox.x + ai_match.bbox.w)
                by2 = max(cv_det.bbox.y + cv_det.bbox.h, ai_match.bbox.y + ai_match.bbox.h)

                c_ai = ai_match.confidence
                c_bright = min(1.0, cv_det.peak_intensity / 255.0)
                c_shape = min(1.0, cv_det.circularity)

                # Temporal score
                if self.last_track_pos is not None:
                    t_dist = math.hypot(
                        u_fused - self.last_track_pos[0],
                        v_fused - self.last_track_pos[1],
                    )
                    c_temp = math.exp(-0.5 * (t_dist / self.temp_sigma) ** 2)
                else:
                    c_temp = 0.5

                confidence = (
                    self.w_ai * c_ai
                    + self.w_bright * c_bright
                    + self.w_shape * c_shape
                    + self.w_temp * c_temp
                )

                fused_candidates.append(
                    Detection(
                        center_u=u_fused,
                        center_v=v_fused,
                        bbox=BoundingBox(x=bx1, y=by1, w=bx2 - bx1, h=by2 - by1),
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        area=float(cv_det.area),
                        peak_intensity=cv_det.peak_intensity,
                        circularity=cv_det.circularity,
                        detector_name="hybrid_fusion_matched",
                        timestamp=timestamp,
                    )
                )
            else:
                # Classical-only candidate (AI missed or model absent)
                c_ai = 0.0
                c_bright = min(1.0, cv_det.peak_intensity / 255.0)
                c_shape = min(1.0, cv_det.circularity)

                if self.last_track_pos is not None:
                    t_dist = math.hypot(
                        cv_det.center_u - self.last_track_pos[0],
                        cv_det.center_v - self.last_track_pos[1],
                    )
                    c_temp = math.exp(-0.5 * (t_dist / self.temp_sigma) ** 2)
                else:
                    c_temp = 0.5

                # Weight redistribution if AI is absent
                if ai_result.success:
                    confidence = (
                        self.w_ai * 0.0
                        + self.w_bright * c_bright
                        + self.w_shape * c_shape
                        + self.w_temp * c_temp
                    )
                else:
                    # When AI is not active, rescale remaining weights to 1.0
                    w_norm = self.w_bright + self.w_shape + self.w_temp
                    confidence = (
                        (self.w_bright * c_bright + self.w_shape * c_shape + self.w_temp * c_temp)
                        / max(1e-5, w_norm)
                    )

                fused_candidates.append(
                    Detection(
                        center_u=cv_det.center_u,
                        center_v=cv_det.center_v,
                        bbox=cv_det.bbox,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        area=cv_det.area,
                        peak_intensity=cv_det.peak_intensity,
                        circularity=cv_det.circularity,
                        detector_name="hybrid_cv_only",
                        timestamp=timestamp,
                    )
                )

        # 2. Add remaining AI-only candidates
        for idx, ai_det in enumerate(ai_candidates):
            if idx not in matched_ai_indices:
                c_ai = ai_det.confidence
                c_bright = 0.5
                c_shape = 0.5

                if self.last_track_pos is not None:
                    t_dist = math.hypot(
                        ai_det.center_u - self.last_track_pos[0],
                        ai_det.center_v - self.last_track_pos[1],
                    )
                    c_temp = math.exp(-0.5 * (t_dist / self.temp_sigma) ** 2)
                else:
                    c_temp = 0.5

                confidence = (
                    self.w_ai * c_ai
                    + self.w_bright * c_bright
                    + self.w_shape * c_shape
                    + self.w_temp * c_temp
                ) * 0.8  # Slight penalty for lack of CV contour

                fused_candidates.append(
                    Detection(
                        center_u=ai_det.center_u,
                        center_v=ai_det.center_v,
                        bbox=ai_det.bbox,
                        confidence=float(np.clip(confidence, 0.0, 1.0)),
                        area=ai_det.area,
                        peak_intensity=180.0,
                        circularity=0.8,
                        detector_name="hybrid_ai_only",
                        timestamp=timestamp,
                    )
                )

        # 3. Select primary beacon based on highest fused confidence
        primary: Optional[Detection] = None
        if fused_candidates:
            fused_candidates.sort(key=lambda d: d.confidence, reverse=True)
            primary = fused_candidates[0]
            self.last_track_pos = (primary.center_u, primary.center_v)
        else:
            self.last_track_pos = None

        total_latency = (
            classical_result.latency_ms
            + ai_result.latency_ms
            + (time.perf_counter() - start_t) * 1000.0
        )

        return DetectionResult(
            primary=primary,
            candidates=fused_candidates,
            latency_ms=total_latency,
            success=(primary is not None),
        )
