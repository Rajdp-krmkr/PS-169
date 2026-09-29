"""
Training pipeline for YOLO optical beacon detector.

Usage:
  py ai/train.py --data datasets/data.yaml --epochs 30 --imgsz 640
"""

from __future__ import annotations
import argparse
import os
import sys

# Ensure root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def train_yolo(
    data_yaml: str = "datasets/data.yaml",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    model_name: str = "yolov8n.pt",
    output_dir: str = "models",
) -> None:
    """Train YOLO nano model on optical beacon dataset."""
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Dataset configuration '{data_yaml}' not found.")
        print("Run ai/dataset_generator.py first to synthesize training samples.")
        return

    print("=" * 64)
    print(" INITIATING YOLO OPTICAL BEACON DETECTOR TRAINING ")
    print(f" Dataset : {data_yaml}")
    print(f" Epochs  : {epochs} | Image Size: {imgsz} | Base: {model_name}")
    print("=" * 64)

    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=output_dir,
            name="beacon_detector",
            exist_ok=True,
        )
        print("\n[SUCCESS] Model training complete.")
        # Export to ONNX for fast zero-dependency inference via cv2.dnn
        onnx_path = model.export(format="onnx")
        print(f"[SUCCESS] Model exported to ONNX format: {onnx_path}")
    except Exception as e:
        print(f"[NOTICE] PyTorch / Ultralytics execution encountered: {e}")
        print("[NOTICE] On this host, Classical CV and Hybrid Detection operate as the primary high-speed engines.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Optical Beacon Detector")
    parser.add_argument("--data", type=str, default="datasets/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    train_yolo(data_yaml=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == "__main__":
    main()
