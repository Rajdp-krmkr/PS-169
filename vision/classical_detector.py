"""
Classical Computer Vision optical beacon detector for FSOC PAT.

Pipeline:
Frame -> Grayscale -> Noise Reduction -> Dynamic/Fixed Thresholding
      -> Morphological Filtering -> Connected Contours -> Feature Extraction
      -> Multi-Criteria Candidate Scoring -> Temporal Association -> Output Detection
"""

from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from vision.detection_types import Detection, BoundingBox, DetectionResult
from vision.preprocessing import preprocess_frame


class ClassicalBeaconDetector:
    """
    High-performance, physics-informed classical CV detector for optical beacons.
    """

    def __init__(
        self,
        binary_threshold: int = 150,
        min_area: float = 4.0,
        max_area: float = 1200.0,
        min_circularity: float = 0.50,
        min_peak_intensity: float = 120.0,
        use_adaptive_threshold: bool = False,
    ) -> None:
        self.threshold = binary_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.min_peak_intensity = min_peak_intensity
        self.use_adaptive = use_adaptive_threshold

        # Morphological structuring element
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Temporal memory for continuity tracking
        self.last_detection_pos: Optional[Tuple[float, float]] = None

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """
        Run classical CV beacon extraction pipeline.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocessing
        gray, _ = preprocess_frame(frame, gaussian_kernel=3)

        # Step 2: Thresholding
        if self.use_adaptive:
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,
                -5,
            )
        else:
            _, binary = cv2.threshold(
                gray, self.threshold, 255, cv2.THRESH_BINARY
            )

        # Step 3: Morphological filtering
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel_open)

        # Step 4: Contour extraction
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[Detection] = []

        # Step 5: Feature extraction & Filtering
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Circularity metric: 4*pi*area / perimeter^2
            circularity = (4.0 * math.pi * area) / (perimeter**2)
            if circularity < self.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = min(w, h) / max(w, h)
            if aspect_ratio < 0.4:
                continue

            # Centroid calculation using spatial moments
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            # Peak intensity within bounding box
            roi = gray[y : y + h, x : x + w]
            peak_val = float(np.max(roi)) if roi.size > 0 else 0.0

            if peak_val < self.min_peak_intensity:
                continue

            # Confidence scoring
            c_circ = min(1.0, circularity)
            c_bright = min(1.0, peak_val / 255.0)
            c_aspect = aspect_ratio
            confidence = 0.45 * c_circ + 0.35 * c_bright + 0.20 * c_aspect

            candidates.append(
                Detection(
                    center_u=cx,
                    center_v=cy,
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    area=float(area),
                    peak_intensity=peak_val,
                    circularity=float(circularity),
                    detector_name="classical_cv",
                    timestamp=timestamp,
                )
            )

        # Step 6: Candidate selection (Primary beacon)
        primary: Optional[Detection] = None
        if candidates:
            if self.last_detection_pos is not None:
                # Score candidates factoring proximity to last known position
                def score_candidate(cand: Detection) -> float:
                    dist = math.hypot(
                        cand.center_u - self.last_detection_pos[0],
                        cand.center_v - self.last_detection_pos[1],
                    )
                    # Gaussian distance weighting (sigma ~ 150px)
                    dist_weight = math.exp(-0.5 * (dist / 150.0) ** 2)
                    return 0.6 * cand.confidence + 0.4 * dist_weight

                candidates.sort(key=score_candidate, reverse=True)
            else:
                # Rank primarily by confidence and brightness
                candidates.sort(
                    key=lambda c: (c.confidence, c.peak_intensity), reverse=True
                )

            primary = candidates[0]
            self.last_detection_pos = (primary.center_u, primary.center_v)
        else:
            # Decay memory if no detection
            self.last_detection_pos = None

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return DetectionResult(
            primary=primary,
            candidates=candidates,
            latency_ms=latency_ms,
            success=(primary is not None),
        )
