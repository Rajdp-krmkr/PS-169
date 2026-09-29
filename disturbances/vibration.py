"""
Platform vibration and angular jitter disturbance models for optical gimbals.

Simulates multi-harmonic engine/rotor vibration and high-frequency stochastic jitter.
"""

from __future__ import annotations
import math
from typing import Tuple, List
import numpy as np

from simulation.coordinate_system import deg2rad


class PlatformVibration:
    """
    Simulates base platform vibrations (e.g. UAV rotor wash, satellite reaction wheels).
    """

    def __init__(
        self,
        base_amplitude_deg: float = 0.15,
        base_frequency_hz: float = 12.0,
        jitter_std_deg: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.base_amp_rad = deg2rad(base_amplitude_deg)
        self.base_freq = base_frequency_hz
        self.jitter_std_rad = deg2rad(jitter_std_deg)
        self.rng = np.random.default_rng(seed)

        # Harmonics [freq_factor, amp_factor, phase_pan, phase_tilt]
        self.harmonics = [
            (1.0, 1.0, 0.0, math.pi / 3.0),
            (2.2, 0.45, 1.2, 0.5),
            (4.8, 0.20, 2.5, 1.9),
        ]

    def sample(
        self, t: float, strength: float = 1.0
    ) -> Tuple[float, float]:
        """
        Sample vibration angular offset (delta_pan_rad, delta_tilt_rad) at time t.
        """
        if strength <= 0.0:
            return 0.0, 0.0

        amp = self.base_amp_rad * strength
        jitter_sigma = self.jitter_std_rad * strength

        vib_pan = 0.0
        vib_tilt = 0.0

        for freq_mult, amp_mult, phi_p, phi_t in self.harmonics:
            omega = 2.0 * math.pi * (self.base_freq * freq_mult)
            vib_pan += (amp * amp_mult) * math.sin(omega * t + phi_p)
            vib_tilt += (amp * amp_mult) * math.cos(omega * t + phi_t)

        # Add Gaussian random high-frequency jitter
        if jitter_sigma > 0.0:
            vib_pan += float(self.rng.normal(0.0, jitter_sigma))
            vib_tilt += float(self.rng.normal(0.0, jitter_sigma))

        return vib_pan, vib_tilt
