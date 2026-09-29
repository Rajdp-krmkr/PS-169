"""
Adaptive Hybrid Kalman <-> Particle Filter Tracker for FSOC Coarse PAT.

Combines:
- Continuous-Discrete Kalman Filter (optimal estimator under Gaussian noise / calm sky).
- Bootstrap Particle Filter (robust non-Gaussian estimator under severe Kolmogorov
  scintillation fades and multi-modal candidate ambiguities).
- Severity Score & Hysteresis Switching Controller:
  - Switches to Particle Filter when severity S > 0.55 or consecutive misses surge.
  - Switches back to Kalman Filter only after M=5 consecutive calm frames (S < 0.30)
    to completely eliminate mode-switching chatter/flickering.
- Bi-directional state handoff (seeding particle cloud from KF covariance, and
  seeding KF from particle cloud mean upon recovery).
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from tracking.kalman import KalmanTracker
from tracking.particle import ParticleTracker
from tracking.severity import OpticalSeverityCalculator
from simulation.coordinate_system import rad2deg


@dataclass
class HybridEstimate:
    az_rad: float
    el_rad: float
    vx_rad_s: float
    vy_rad_s: float
    active_filter: str  # "KF" or "PF"
    severity_score: float
    mode_switched: bool


class HybridPATTracker:
    """
    Seamless hybrid estimator dynamically selecting between Kalman Filter
    and Particle Filter.
    """

    def __init__(
        self,
        severity_switch_to_pf: float = 0.55,
        severity_switch_to_kf: float = 0.30,
        consecutive_calm_to_kf: int = 5,
        q_process_spectral: float = 0.01,
        r_measurement_std_rad: float = 0.002,
        num_particles: int = 250,
        seed: int = 42,
    ) -> None:
        self.thresh_to_pf = severity_switch_to_pf
        self.thresh_to_kf = severity_switch_to_kf
        self.calm_needed = consecutive_calm_to_kf

        self.kalman = KalmanTracker(
            q_process_spectral=q_process_spectral,
            r_measurement_std=r_measurement_std_rad,
        )
        self.particle = ParticleTracker(
            num_particles=num_particles,
            meas_pos_std=r_measurement_std_rad,
            seed=seed,
        )
        self.severity_calc = OpticalSeverityCalculator()

        self.active_mode: str = "KF"  # "KF" or "PF"
        self.consecutive_calm_frames: int = 0
        self.initialized = False
        self.total_switches: int = 0

    def initialize(self, az: float, el: float) -> None:
        """Initialize both internal estimators."""
        self.kalman.initialize(az, el)
        self.particle.initialize(az, el)
        self.initialized = True
        self.active_mode = "KF"
        self.consecutive_calm_frames = 0

    def reset(self) -> None:
        """Reset hybrid estimator."""
        self.kalman.reset()
        self.particle.reset()
        self.severity_calc.reset()
        self.active_mode = "KF"
        self.consecutive_calm_frames = 0
        self.initialized = False

    def step(
        self,
        dt: float,
        meas_az: Optional[float],
        meas_el: Optional[float],
        confidence: float = 1.0,
        turbulence_level: float = 0.0,
        is_occluded: bool = False,
        consecutive_misses: int = 0,
    ) -> HybridEstimate:
        """
        Advance state estimation by dt, compute channel severity, and apply
        hysteresis-governed mode switching.
        """
        if not self.initialized:
            if meas_az is not None and meas_el is not None:
                self.initialize(meas_az, meas_el)
            return HybridEstimate(0.0, 0.0, 0.0, 0.0, self.active_mode, 0.0, False)

        # 1. State propagation step
        kf_prior = self.kalman.predict(dt)
        pf_prior = self.particle.predict(dt)

        # 2. Measurement update step
        if meas_az is not None and meas_el is not None:
            self.kalman.update((meas_az, meas_el))
            self.particle.update(meas_az, meas_el, confidence=confidence)
            curr_pos = (self.kalman.x[0], self.kalman.x[1])
            tracking_err = math.hypot(meas_az - curr_pos[0], meas_el - curr_pos[1])
        else:
            tracking_err = 0.005

        # 3. Channel Severity Assessment
        severity = self.severity_calc.compute(
            confidence=confidence if meas_az is not None else 0.0,
            tracking_error_rad=tracking_err,
            turbulence_level=turbulence_level,
            is_occluded=is_occluded,
            consecutive_misses=consecutive_misses,
        )

        # 4. Hysteresis Mode Switching Logic
        mode_switched = False
        if self.active_mode == "KF":
            if severity >= self.thresh_to_pf or consecutive_misses >= 3:
                # Switch to Particle Filter
                self.active_mode = "PF"
                mode_switched = True
                self.total_switches += 1
                self.consecutive_calm_frames = 0
                # Seed particle filter from current KF state
                self.particle.initialize(
                    self.kalman.x[0],
                    self.kalman.x[1],
                    self.kalman.x[2],
                    self.kalman.x[3],
                )
        elif self.active_mode == "PF":
            if severity <= self.thresh_to_kf and not is_occluded and consecutive_misses == 0:
                self.consecutive_calm_frames += 1
                if self.consecutive_calm_frames >= self.calm_needed:
                    # Switch back to Kalman Filter
                    self.active_mode = "KF"
                    mode_switched = True
                    self.total_switches += 1
                    self.consecutive_calm_frames = 0
                    # Seed KF from current particle cloud estimate
                    p_state = self.particle.get_state()
                    self.kalman.x = np.array(p_state, dtype=np.float64)
            else:
                self.consecutive_calm_frames = 0

        # 5. Extract output state from active estimator
        if self.active_mode == "KF":
            out_az = float(self.kalman.x[0])
            out_el = float(self.kalman.x[1])
            out_vx = float(self.kalman.x[2])
            out_vy = float(self.kalman.x[3])
        else:
            p_state = self.particle.get_state()
            out_az, out_el, out_vx, out_vy = p_state

        return HybridEstimate(
            az_rad=out_az,
            el_rad=out_el,
            vx_rad_s=out_vx,
            vy_rad_s=out_vy,
            active_filter=self.active_mode,
            severity_score=severity,
            mode_switched=mode_switched,
        )
