"""
AI & Classical Detector Inference Benchmark Tool.

Compares:
1. Classical CV Detector
2. AI YOLO Detector
3. Multi-Modal Hybrid Fusion
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.classical_detector import ClassicalBeaconDetector
from vision.yolo_detector import YOLODetector
from vision.fusion import DetectionFusion
from simulation.beacon import Beacon


def benchmark_detectors(num_runs: int = 50) -> None:
    """Run comparative benchmark between Classical, AI, and Hybrid detection."""
    print("=" * 68)
    print(" OPTICAL BEACON DETECTOR COMPARATIVE LATENCY & ACCURACY BENCHMARK ")
    print("=" * 68)

    # Create synthetic test frame
    frame = np.full((720, 1280, 3), 15, dtype=np.uint8)
    beacon = Beacon(beacon_id=1, intensity=250.0, radius_pixels=7.0)
    beacon.render_to_frame(frame, 640.0, 360.0)

    # Initialize detectors
    classical = ClassicalBeaconDetector(binary_threshold=140)
    yolo = YOLODetector(model_path="models/beacon_detector.onnx")
    fusion = DetectionFusion()

    # 1. Benchmark Classical
    latencies_cv = []
    for _ in range(num_runs):
        res = classical.detect(frame)
        latencies_cv.append(res.latency_ms)

    # 2. Benchmark AI (or fallback)
    latencies_ai = []
    for _ in range(num_runs):
        res = yolo.detect(frame)
        latencies_ai.append(res.latency_ms)

    # 3. Benchmark Hybrid
    latencies_hybrid = []
    for _ in range(num_runs):
        res_cv = classical.detect(frame)
        res_ai = yolo.detect(frame)
        res_fused = fusion.fuse(res_cv, res_ai)
        latencies_hybrid.append(res_fused.latency_ms)

    print(f" 1. Classical CV Detector : Avg {np.mean(latencies_cv):.2f} ms ({1000.0/np.mean(latencies_cv):.1f} FPS)")
    print(f" 2. AI YOLO Detector     : Avg {np.mean(latencies_ai):.2f} ms (Loaded: {yolo.is_loaded})")
    print(f" 3. Hybrid Fusion Pipeline: Avg {np.mean(latencies_hybrid):.2f} ms ({1000.0/np.mean(latencies_hybrid):.1f} FPS)")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Detector Benchmark Tool")
    parser.add_argument("--runs", type=int, default=30, help="Number of benchmark iterations")
    args = parser.parse_args()
    benchmark_detectors(num_runs=args.runs)


if __name__ == "__main__":
    main()
