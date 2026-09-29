"""
Unit tests for disturbance models (noise, vibration, turbulence, blur, manager).
"""

import math
import numpy as np
import pytest
import cv2

from disturbances.noise import SensorNoise
from disturbances.vibration import PlatformVibration
from disturbances.turbulence import AtmosphericTurbulence
from disturbances.blur import MotionBlur, DefocusBlur
from disturbances.manager import DisturbanceManager
from simulation.camera import VirtualCamera


def test_sensor_noise():
    noise = SensorNoise(gaussian_std=25.0, seed=42)
    clean_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    noisy_frame = noise.apply(clean_frame, strength=1.0)
    assert noisy_frame.shape == clean_frame.shape
    assert noisy_frame.dtype == np.uint8

    # Noise should alter pixels around mean 128
    std_diff = float(np.std(noisy_frame.astype(np.float32) - 128.0))
    assert 15.0 < std_diff < 35.0


def test_platform_vibration():
    vib = PlatformVibration(base_amplitude_deg=0.5, base_frequency_hz=10.0, seed=42)
    pan1, tilt1 = vib.sample(t=0.0, strength=1.0)
    pan2, tilt2 = vib.sample(t=0.025, strength=1.0)  # Quarter period

    assert abs(pan1 - pan2) > 1e-4 or abs(tilt1 - tilt2) > 1e-4
    # Zero strength produces zero vibration
    p0, t0 = vib.sample(t=1.0, strength=0.0)
    assert p0 == 0.0 and t0 == 0.0


def test_atmospheric_turbulence():
    turb = AtmosphericTurbulence(resolution=(200, 200), max_displacement_pixels=5.0, seed=42)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame, (100, 100), 20, (255, 255, 255), -1)

    warped = turb.apply(frame, sim_time=1.0, strength=1.0)
    assert warped.shape == frame.shape
    assert np.all(np.isfinite(warped))
    assert np.max(warped) > 200


def test_disturbance_manager_coordination():
    mgr = DisturbanceManager(resolution=(640, 480), seed=42)
    camera = VirtualCamera(resolution=(640, 480))

    mgr.set_strengths(vibration=50.0, noise=25.0, turbulence=30.0)
    assert math.isclose(mgr.vibration_strength, 0.5, abs_tol=1e-5)
    assert math.isclose(mgr.noise_strength, 0.25, abs_tol=1e-5)

    # Step optical vibration
    mgr.step_optical(camera, sim_time=0.5)
    assert camera.pan_jitter_offset != 0.0 or camera.tilt_jitter_offset != 0.0

    # Apply visual distortions
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    disturbed, dropped = mgr.apply_visual(frame, camera, sim_time=0.5)
    assert disturbed.shape == (480, 640, 3)
    assert dropped is False
