"""
Sequential Importance Resampling (SIR) Particle Filter for FSOC Optical Tracking.

Under severe atmospheric turbulence (Kolmogorov Cn^2 >= 1e-13 m^-2/3) and deep
scintillation fades, measurement noise distributions become strongly non-Gaussian
and multi-modal. A Linear Kalman Filter can diverge or lose track during deep fades.

The Particle Filter represents the posterior distribution of the target state
using N discrete weighted particles:
    x_i = [az, el, vx, vy]^T,   w_i in [0, 1],   sum(w_i) = 1.0

Features:
- Non-Gaussian observation likelihood with robust heavy-tailed measurement model.
- Systematic resampling triggered by Effective Sample Size (N_eff < N/2).
- Trajectory extrapolation during measurement dropouts (coasting).
"""

from __future__ import annotations
import math
from typing import Tuple, Optional
import numpy as np


class ParticleTracker:
    """
    Bootstrap Particle Filter for 2-axis optical angular target tracking.
    """

    def __init__(
        self,
        num_particles: int = 300,
        process_pos_std: float = 0.001,   # ~0.057 deg
        process_vel_std: float = 0.005,   # ~0.28 deg/s
        meas_pos_std: float = 0.002,      # ~0.11 deg
        seed: int = 42,
    ) -> None:
        self.num_particles = max(50, int(num_particles))
        self.q_pos_std = float(process_pos_std)
        self.q_vel_std = float(process_vel_std)
        self.r_std = float(meas_pos_std)
        self.rng = np.random.default_rng(seed)

        # Particles shape: (N, 4) -> [az, el, vx, vy]
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float64)
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.initialized = False
        self.estimated_state = np.zeros(4, dtype=np.float64)

    def initialize(self, az: float, el: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Initialize particle cloud around initial position."""
        self.particles[:, 0] = az + self.rng.normal(0.0, self.q_pos_std * 2.0, self.num_particles)
        self.particles[:, 1] = el + self.rng.normal(0.0, self.q_pos_std * 2.0, self.num_particles)
        self.particles[:, 2] = vx + self.rng.normal(0.0, self.q_vel_std, self.num_particles)
        self.particles[:, 3] = vy + self.rng.normal(0.0, self.q_vel_std, self.num_particles)
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = True
        self.estimated_state = np.array([az, el, vx, vy], dtype=np.float64)

    def reset(self) -> None:
        """Reset particle filter."""
        self.particles.fill(0.0)
        self.weights.fill(1.0 / self.num_particles)
        self.initialized = False
        self.estimated_state.fill(0.0)

    def predict(self, dt: float) -> np.ndarray:
        """
        Kinematic propagation of particles through constant-velocity dynamic model with noise.
        """
        if not self.initialized:
            return self.estimated_state.copy()

        # Propagate positions: x = x + vx * dt + noise
        self.particles[:, 0] += self.particles[:, 2] * dt + self.rng.normal(
            0.0, self.q_pos_std * math.sqrt(dt * 30.0), self.num_particles
        )
        self.particles[:, 1] += self.particles[:, 3] * dt + self.rng.normal(
            0.0, self.q_pos_std * math.sqrt(dt * 30.0), self.num_particles
        )
        # Propagate velocities: vx = vx + noise
        self.particles[:, 2] += self.rng.normal(0.0, self.q_vel_std * math.sqrt(dt * 30.0), self.num_particles)
        self.particles[:, 3] += self.rng.normal(0.0, self.q_vel_std * math.sqrt(dt * 30.0), self.num_particles)

        # Update mean state estimate
        self.estimated_state = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        return self.estimated_state.copy()

    def update(
        self,
        meas_az: Optional[float],
        meas_el: Optional[float],
        confidence: float = 1.0,
    ) -> np.ndarray:
        """
        Weight update given optical measurement.
        If meas is None or confidence < 0.20, coasts on particle cloud.
        """
        if not self.initialized:
            if meas_az is not None and meas_el is not None:
                self.initialize(meas_az, meas_el)
            return self.estimated_state.copy()

        if meas_az is None or meas_el is None or confidence < 0.20:
            # Dropout/fade: particles diffuse without measurement update
            return self.estimated_state.copy()

        # Robust likelihood: Gaussian core + heavy Student-t / uniform tails for scintillation fades
        dx = self.particles[:, 0] - meas_az
        dy = self.particles[:, 1] - meas_el
        dist_sq = dx**2 + dy**2

        # Effective measurement variance scaled inversely with detector confidence
        eff_var = (self.r_std / max(0.2, confidence))**2
        gaussian_likelihood = np.exp(-0.5 * dist_sq / eff_var)
        heavy_tail_likelihood = 0.05 / (1.0 + dist_sq / eff_var)

        likelihood = 0.95 * gaussian_likelihood + heavy_tail_likelihood
        self.weights *= likelihood
        weight_sum = np.sum(self.weights)

        if weight_sum > 1e-12:
            self.weights /= weight_sum
        else:
            # Cloud collapsed or extreme outlier -> re-weight uniformly
            self.weights.fill(1.0 / self.num_particles)

        # Compute updated state estimate
        self.estimated_state = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)

        # Resample if effective sample size drops below N/2
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.num_particles / 2.0:
            self._systematic_resample()

        return self.estimated_state.copy()

    def _systematic_resample(self) -> None:
        """Low-variance systematic resampling."""
        cdf = np.cumsum(self.weights)
        u0 = self.rng.uniform(0.0, 1.0 / self.num_particles)
        u = u0 + np.arange(self.num_particles) / self.num_particles

        indexes = np.zeros(self.num_particles, dtype=int)
        i, j = 0, 0
        while i < self.num_particles:
            if u[i] < cdf[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        self.particles = self.particles[indexes].copy()
        self.weights.fill(1.0 / self.num_particles)

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return (az, el, vx, vy)."""
        return (
            float(self.estimated_state[0]),
            float(self.estimated_state[1]),
            float(self.estimated_state[2]),
            float(self.estimated_state[3]),
        )
