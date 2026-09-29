"""
Unified Disturbance Manager for FSOC PAT simulator.

Coordinates:
- Sensor noise (Gaussian & Salt-and-Pepper)
- Platform vibration & mechanical angular jitter
- Visual atmospheric turbulence
- Directional motion blur & optical defocus
- Artificial target occlusion and dropped frames
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple, Optional
import numpy as np

from disturbances.noise import SensorNoise
from disturbances.vibration import PlatformVibration
from disturbances.turbulence import AtmosphericTurbulence
from disturbances.blur import MotionBlur, DefocusBlur
from disturbances.occluder import OccluderManager, DynamicOccluder
from simulation.camera import VirtualCamera


class DisturbanceManager:
    """
    Central disturbance control hub supporting 0-100% normalized sliders.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        seed: int = 42,
    ) -> None:
        self.sensor_noise = SensorNoise(gaussian_std=20.0, salt_pepper_prob=0.005, seed=seed)
        self.vibration = PlatformVibration(
            base_amplitude_deg=0.25,
            base_frequency_hz=14.0,
            jitter_std_deg=0.03,
            seed=seed,
        )
        self.turbulence = AtmosphericTurbulence(
            resolution=resolution,
            max_displacement_pixels=10.0,
            temporal_frequency_hz=1.8,
            seed=seed,
        )
        self.motion_blur = MotionBlur(max_kernel_size=17)
        self.defocus = DefocusBlur(max_sigma=3.5)
        self.occluders = OccluderManager()

        # Normalized strengths [0.0 to 1.0]
        self.noise_strength: float = 0.0
        self.vibration_strength: float = 0.0
        self.turbulence_strength: float = 0.0
        self.motion_blur_strength: float = 0.0
        self.defocus_strength: float = 0.0
        self.frame_drop_prob: float = 0.0

        # Artificial occlusion toggle for target loss/reacquisition evaluation
        self.occlusion_active: bool = False

        self.rng = np.random.default_rng(seed)

    def set_strengths(
        self,
        noise: Optional[float] = None,
        vibration: Optional[float] = None,
        turbulence: Optional[float] = None,
        motion_blur: Optional[float] = None,
        defocus: Optional[float] = None,
        frame_drop: Optional[float] = None,
        occlusion: Optional[bool] = None,
    ) -> None:
        """Set normalized disturbance levels (0.0 to 1.0 or 0 to 100%)."""
        if noise is not None:
            self.noise_strength = float(np.clip(noise / 100.0 if noise > 1.0 else noise, 0.0, 1.0))
        if vibration is not None:
            self.vibration_strength = float(np.clip(vibration / 100.0 if vibration > 1.0 else vibration, 0.0, 1.0))
        if turbulence is not None:
            self.turbulence_strength = float(np.clip(turbulence / 100.0 if turbulence > 1.0 else turbulence, 0.0, 1.0))
        if motion_blur is not None:
            self.motion_blur_strength = float(np.clip(motion_blur / 100.0 if motion_blur > 1.0 else motion_blur, 0.0, 1.0))
        if defocus is not None:
            self.defocus_strength = float(np.clip(defocus / 100.0 if defocus > 1.0 else defocus, 0.0, 1.0))
        if frame_drop is not None:
            self.frame_drop_prob = float(np.clip(frame_drop / 100.0 if frame_drop > 1.0 else frame_drop, 0.0, 1.0))
        if occlusion is not None:
            self.occlusion_active = bool(occlusion)

    def step_optical(self, camera: VirtualCamera, sim_time: float) -> None:
        """
        Inject base platform vibration into camera line-of-sight pointing.
        """
        vib_pan, vib_tilt = self.vibration.sample(
            sim_time, strength=self.vibration_strength
        )
        camera.pan_jitter_offset = vib_pan
        camera.tilt_jitter_offset = vib_tilt

    def apply_visual(
        self,
        frame: np.ndarray,
        camera: VirtualCamera,
        sim_time: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        Apply visual distortions (turbulence, motion blur, defocus, noise).
        Returns: (disturbed_frame, frame_was_dropped)
        """
        # Random frame drop simulation
        if self.frame_drop_prob > 0.0 and self.rng.uniform(0.0, 1.0) < self.frame_drop_prob:
            return frame, True

        out = frame

        # 1. Atmospheric Turbulence
        if self.turbulence_strength > 0.0:
            out = self.turbulence.apply(out, sim_time, strength=self.turbulence_strength)

        # 2. Motion Blur
        if self.motion_blur_strength > 0.0:
            out = self.motion_blur.apply(
                out,
                camera.pan_velocity,
                camera.tilt_velocity,
                strength=self.motion_blur_strength,
            )

        # 3. Optical Defocus Blur
        if self.defocus_strength > 0.0:
            out = self.defocus.apply(out, strength=self.defocus_strength)

        # 4. Sensor Noise
        if self.noise_strength > 0.0:
            out = self.sensor_noise.apply(out, strength=self.noise_strength)

        # 5. Dynamic Obstacles / Cloud Occluders
        out = self.occluders.render_all(camera, out)

        return out, False

    def step_occluders(self, dt: float) -> None:
        """Advance dynamic physical occluder positions."""
        self.occluders.step(dt)

    def check_beacon_occlusion(self, target_az_rad: float, target_el_rad: float) -> Tuple[bool, float]:
        """
        Check if target line-of-sight is obstructed by artificial toggle or physical occluder.
        Returns: (is_occluded_bool, transmission_factor)
        """
        is_phys_occ, factor = self.occluders.check_occlusion(target_az_rad, target_el_rad)
        if self.occlusion_active:
            return True, 0.0
        return is_phys_occ, factor
