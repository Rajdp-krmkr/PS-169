"""
Unit tests for synthetic dataset generator.
"""

import os
import shutil
import pytest
import cv2
from ai.dataset_generator import SyntheticDatasetGenerator


def test_dataset_generation():
    test_out_dir = "datasets/test_synthetic_dataset"
    generator = SyntheticDatasetGenerator(
        output_dir=test_out_dir,
        resolution=(320, 240),
        seed=100,
    )

    try:
        res = generator.generate(num_train=6, num_val=3, clean_first=True)
        assert os.path.exists(res["data_yaml"])

        # Check training images
        train_img_dir = os.path.join(test_out_dir, "images", "train")
        train_lbl_dir = os.path.join(test_out_dir, "labels", "train")
        train_imgs = os.listdir(train_img_dir)
        train_lbls = os.listdir(train_lbl_dir)

        assert len(train_imgs) == 6
        assert len(train_lbls) == 6

        # Check an image and its label
        first_img = cv2.imread(os.path.join(train_img_dir, train_imgs[0]))
        assert first_img is not None
        assert first_img.shape == (240, 320, 3)

        with open(os.path.join(train_lbl_dir, train_lbls[0]), "r") as f:
            lines = f.readlines()
            assert len(lines) >= 1
            parts = [float(p) for p in lines[0].strip().split()]
            assert len(parts) == 5
            # Class ID is 0
            assert parts[0] == 0
            # Normalized coordinates between 0 and 1
            assert 0.0 < parts[1] < 1.0  # x_center
            assert 0.0 < parts[2] < 1.0  # y_center
            assert 0.0 < parts[3] < 1.0  # width
            assert 0.0 < parts[4] < 1.0  # height
    finally:
        if os.path.exists(test_out_dir):
            shutil.rmtree(test_out_dir)
