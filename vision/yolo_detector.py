"""
AI Beacon Detector supporting Ultralytics YOLO and OpenCV DNN (ONNX) inference.

Features:
- Optional AI detection with graceful fallback to Classical CV if model is absent
- High-speed zero-dependency inference via cv2.dnn for ONNX models
- Automatic image preprocessing, letterbox scaling, and NMS postprocessing
- Non-blocking error handling
"""

from __future__ import annotations
import math
import os
import time
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from vision.detection_types import Detection, BoundingBox, DetectionResult


class YOLODetector:
    """
    Lightweight neural network detector for optical beacons.
    Supports .onnx (via OpenCV DNN) and .pt (via Ultralytics).
    """

    def __init__(
        self,
        model_path: Optional[str] = "models/beacon_detector.onnx",
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        self.is_loaded = False
        self.backend: Optional[str] = None
        self.net: Any = None
        self.ultralytics_model: Any = None

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"[INFO] AI Detector: Model file '{model_path}' not found. Operating in fallback mode.")

    def _load_model(self, path: str) -> bool:
        """Attempt to load neural network weights."""
        ext = os.path.splitext(path)[1].lower()

        # 1. OpenCV DNN for ONNX models (Zero external dependencies)
        if ext == ".onnx":
            try:
                self.net = cv2.dnn.readNetFromONNX(path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.is_loaded = True
                self.backend = "opencv_dnn"
                print(f"[INFO] AI Detector loaded successfully via OpenCV DNN: {path}")
                return True
            except Exception as e:
                print(f"[WARN] Failed to load ONNX model via cv2.dnn: {e}")

        # 2. Ultralytics YOLO for .pt models
        elif ext in (".pt", ".yaml"):
            try:
                from ultralytics import YOLO
                self.ultralytics_model = YOLO(path)
                self.is_loaded = True
                self.backend = "ultralytics"
                print(f"[INFO] AI Detector loaded successfully via Ultralytics: {path}")
                return True
            except Exception as e:
                print(f"[WARN] Failed to load PyTorch/Ultralytics model: {e}")

        return False

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """
        Run inference on frame. Returns DetectionResult.
        """
        if not self.is_loaded:
            return DetectionResult(primary=None, candidates=[], latency_ms=0.0, success=False)

        start_t = time.perf_counter()

        if self.backend == "opencv_dnn":
            result = self._infer_opencv_dnn(frame, timestamp)
        elif self.backend == "ultralytics":
            result = self._infer_ultralytics(frame, timestamp)
        else:
            result = DetectionResult(primary=None, candidates=[], latency_ms=0.0, success=False)

        result.latency_ms = (time.perf_counter() - start_t) * 1000.0
        return result

    def _infer_opencv_dnn(self, frame: np.ndarray, timestamp: float) -> DetectionResult:
        """Inference using OpenCV DNN module on ONNX model."""
        h, w = frame.shape[:2]
        iw, ih = self.input_size

        # Create input blob (scale 1/255, swapRB=True)
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (iw, ih), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Parse standard YOLOv8 output tensor: [1, 5, 8400] (x_c, y_c, w, h, class_score)
        predictions = outputs[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # Transpose to [8400, 5]

        boxes = []
        confidences = []

        scale_x = w / iw
        scale_y = h / ih

        for row in predictions:
            score = float(row[4])
            if score >= self.conf_threshold:
                bx_c = float(row[0]) * scale_x
                by_c = float(row[1]) * scale_y
                bw = float(row[2]) * scale_x
                bh = float(row[3]) * scale_y
                bx = int(bx_c - bw / 2.0)
                by = int(by_c - bh / 2.0)

                boxes.append([bx, by, int(bw), int(bh)])
                confidences.append(score)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        candidates: List[Detection] = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                conf = confidences[i]
                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                candidates.append(
                    Detection(
                        center_u=cx,
                        center_v=cy,
                        bbox=BoundingBox(x=bx, y=by, w=bw, h=bh),
                        confidence=conf,
                        area=float(bw * bh),
                        peak_intensity=255.0,
                        circularity=0.85,
                        detector_name="ai_yolo_onnx",
                        timestamp=timestamp,
                    )
                )

        primary = candidates[0] if candidates else None
        return DetectionResult(
            primary=primary,
            candidates=candidates,
            success=(primary is not None),
        )

    def _infer_ultralytics(self, frame: np.ndarray, timestamp: float) -> DetectionResult:
        """Inference using Ultralytics YOLO instance."""
        results = self.ultralytics_model.predict(
            frame, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False
        )
        candidates: List[Detection] = []

        for r in results:
            for b in r.boxes:
                xywh = b.xywh[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cx, cy, bw, bh = xywh
                bx = int(cx - bw / 2.0)
                by = int(cy - bh / 2.0)

                candidates.append(
                    Detection(
                        center_u=float(cx),
                        center_v=float(cy),
                        bbox=BoundingBox(x=bx, y=by, w=int(bw), h=int(bh)),
                        confidence=conf,
                        area=float(bw * bh),
                        peak_intensity=255.0,
                        circularity=0.90,
                        detector_name="ai_yolo_pt",
                        timestamp=timestamp,
                    )
                )

        primary = candidates[0] if candidates else None
        return DetectionResult(
            primary=primary,
            candidates=candidates,
            success=(primary is not None),
        )
