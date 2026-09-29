"""
Synthetic Optical Beacon Dataset Generator for YOLO Training.

Generates photorealistic optical beacon scenes under varying:
- Boresight positions and trajectory angles
- Background starfield densities and ambient photon gradients
- Sensor noise, platform vibration, motion blur, and atmospheric turbulence
- Clutter, space debris, and optical distractors
- Exports labels in standard YOLO format with data.yaml
"""

from __future__ import annotations
import math
import os
import shutil
from typing import Dict, Any, Tuple, List, Optional
import cv2
import numpy as np

from simulation.coordinate_system import deg2rad
from simulation.beacon import Beacon
from simulation.camera import VirtualCamera
from simulation.environment import VirtualEnvironment
from disturbances.manager import DisturbanceManager


class SyntheticDatasetGenerator:
    """
    Automated synthetic image & label generator compatible with Ultralytics YOLO.
    """

    def __init__(
        self,
        output_dir: str = "datasets",
        resolution: Tuple[int, int] = (640, 480),
        seed: int = 42,
    ) -> None:
        self.output_dir = output_dir
        self.width, self.height = resolution
        self.rng = np.random.default_rng(seed)

        self.images_train_dir = os.path.join(output_dir, "images", "train")
        self.images_val_dir = os.path.join(output_dir, "images", "val")
        self.labels_train_dir = os.path.join(output_dir, "labels", "train")
        self.labels_val_dir = os.path.join(output_dir, "labels", "val")

    def _prepare_directories(self, clean: bool = False) -> None:
        """Create directory structure for YOLO dataset."""
        if clean and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        for d in [
            self.images_train_dir,
            self.images_val_dir,
            self.labels_train_dir,
            self.labels_val_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def generate_single_sample(
        self,
        sample_idx: int,
        camera: VirtualCamera,
        dist_mgr: DisturbanceManager,
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """
        Generate a single randomized synthetic frame with ground-truth YOLO labels.
        Returns: (frame, list_of_labels) where each label is (class_id, x_c, y_c, w, h)
        """
        # 1. Base dark background with ambient photons
        ambient = int(self.rng.integers(5, 25))
        frame = np.full((self.height, self.width, 3), ambient, dtype=np.uint8)

        # 2. Add randomized background stars
        num_stars = int(self.rng.integers(20, 80))
        star_x = self.rng.integers(0, self.width, num_stars)
        star_y = self.rng.integers(0, self.height, num_stars)
        star_bri = self.rng.integers(30, 120, num_stars, dtype=np.uint8)
        for sx, sy, sb in zip(star_x, star_y, star_bri):
            frame[sy, sx] = [sb, sb, sb]

        labels: List[Tuple[int, float, float, float, float]] = []

        # 3. Primary Optical Beacon
        # Place beacon randomly in frame with 5% margin
        margin_x = self.width * 0.08
        margin_y = self.height * 0.08
        u = float(self.rng.uniform(margin_x, self.width - margin_x))
        v = float(self.rng.uniform(margin_y, self.height - margin_y))

        radius = float(self.rng.uniform(4.0, 9.0))
        intensity = float(self.rng.uniform(180.0, 255.0))

        primary_beacon = Beacon(
            beacon_id=1,
            is_primary=True,
            intensity=intensity,
            radius_pixels=radius,
            color_bgr=(255, 255, int(self.rng.integers(200, 255))),
        )
        primary_beacon.render_to_frame(frame, u, v)

        # Compute bounding box in pixels: span is approximately 3 * sigma
        sigma = max(radius / 2.5, 0.8)
        box_half = int(math.ceil(sigma * 3.5)) + 2
        bw = box_half * 2
        bh = box_half * 2

        # YOLO format: normalized [0, 1]
        x_center_norm = u / self.width
        y_center_norm = v / self.height
        w_norm = min(1.0, bw / self.width)
        h_norm = min(1.0, bh / self.height)

        # Class 0 = optical_beacon
        labels.append((0, x_center_norm, y_center_norm, w_norm, h_norm))

        # 4. Optional Optical Distractors (Class 1 or background clutter)
        num_distractors = int(self.rng.integers(0, 3))
        for d_id in range(num_distractors):
            du = float(self.rng.uniform(margin_x, self.width - margin_x))
            dv = float(self.rng.uniform(margin_y, self.height - margin_y))
            # Keep away from primary beacon
            if math.hypot(du - u, dv - v) < 40.0:
                continue

            d_rad = float(self.rng.uniform(2.5, 5.5))
            d_int = float(self.rng.uniform(110.0, 190.0))
            distractor = Beacon(
                beacon_id=10 + d_id,
                is_primary=False,
                intensity=d_int,
                radius_pixels=d_rad,
                color_bgr=(
                    int(self.rng.integers(100, 255)),
                    int(self.rng.integers(100, 200)),
                    int(self.rng.integers(200, 255)),
                ),
            )
            distractor.render_to_frame(frame, du, dv)

        # 5. Apply Randomized Disturbances
        dist_mgr.set_strengths(
            noise=float(self.rng.uniform(0.0, 0.25)),
            turbulence=float(self.rng.uniform(0.0, 0.20)),
            motion_blur=float(self.rng.uniform(0.0, 0.15)),
            defocus=float(self.rng.uniform(0.0, 0.10)),
        )
        sim_t = float(sample_idx * 0.1)
        disturbed_frame, _ = dist_mgr.apply_visual(frame, camera, sim_time=sim_t)

        return disturbed_frame, labels

    def generate(
        self,
        num_train: int = 100,
        num_val: int = 25,
        clean_first: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset and write data.yaml.
        """
        self._prepare_directories(clean=clean_first)
        camera = VirtualCamera(resolution=(self.width, self.height))
        dist_mgr = DisturbanceManager(resolution=(self.width, self.height))

        # Generate Training Set
        for idx in range(num_train):
            frame, labels = self.generate_single_sample(idx, camera, dist_mgr)
            img_filename = f"beacon_train_{idx:05d}.jpg"
            lbl_filename = f"beacon_train_{idx:05d}.txt"

            cv2.imwrite(os.path.join(self.images_train_dir, img_filename), frame)
            with open(os.path.join(self.labels_train_dir, lbl_filename), "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        # Generate Validation Set
        for idx in range(num_val):
            frame, labels = self.generate_single_sample(10000 + idx, camera, dist_mgr)
            img_filename = f"beacon_val_{idx:05d}.jpg"
            lbl_filename = f"beacon_val_{idx:05d}.txt"

            cv2.imwrite(os.path.join(self.images_val_dir, img_filename), frame)
            with open(os.path.join(self.labels_val_dir, lbl_filename), "w") as f:
                for lbl in labels:
                    f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        # Create YOLO data.yaml
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        yaml_content = (
            f"path: {os.path.abspath(self.output_dir)}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: optical_beacon\n"
        )
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        return {
            "dataset_dir": self.output_dir,
            "train_samples": num_train,
            "val_samples": num_val,
            "data_yaml": yaml_path,
        }
