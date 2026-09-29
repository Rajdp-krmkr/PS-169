"""
Unified High-Level PAT Tracker for FSOC Coarse Alignment.

Coordinates:
- Optical measurement processing and coordinate transformations
- Kalman filter state estimation & discrete Wiener process filtering
- Adaptive Hybrid Kalman <-> Particle Filter estimation with scintillation severity
- Trajectory lead-ahead prediction for lag-free gimbal control
- Aerospace 10-state acquisition lifecycle
- Dynamic Safe FOV & Loss-of-Lock early warning risk score
- Fine-PAT Handoff Readiness Engine
- Automatic acquisition and adaptive predictive reacquisition search patterns
"""

from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from tracking.kalman import KalmanTracker
from tracking.particle import ParticleTracker
from tracking.hybrid import HybridPATTracker
from tracking.state_machine import TrackingStateMachine, PATState
from control.search_strategy import SearchStrategy, SpiralScan, RasterScan, PredictiveSearch
from control.safe_fov import DynamicSafeFOVEngine, SafeFOVTelemetry
from control.handoff import FinePATHandoffEngine, HandoffTelemetry
from simulation.coordinate_system import rad2deg
from simulation.camera import VirtualCamera
from vision.detection_types import Detection


@dataclass
class TrackerOutput:
    state: PATState
    state_changed: bool
    # Angular error commanded to controller (rad)
    cmd_delta_az_rad: float
    cmd_delta_el_rad: float
    # Rate command (rad/s) if in search mode, otherwise None
    search_rate_rad_s: Optional[Tuple[float, float]] = None
    # Estimated target velocity (rad/s)
    estimated_vel_rad_s: Tuple[float, float] = (0.0, 0.0)
    # Predicted target position in world radians
    predicted_world_pos_rad: Optional[Tuple[float, float]] = None
    # Projected future pixel on camera sensor
    predicted_pixel_uv: Optional[Tuple[float, float]] = None
    is_locked: bool = False
    # Aerospace Additions:
    active_filter: str = "KF"               # "KF" or "PF"
    severity_score: float = 0.0             # 0.0 to 1.0
    handoff_score_pct: float = 0.0          # 0.0 to 100.0%
    handoff_state: str = "NOT READY"        # "READY", "STABILIZING", "NOT READY"
    risk_state: str = "STABLE"              # "STABLE", "WARNING", "LOCK AT RISK", "CRITICAL"
    risk_score: float = 0.1                 # 0.0 to 1.0
    safe_fov_box_pixels: Optional[Tuple[int, int, int, int]] = None
    dist_to_safe_edge_mrad: float = 10.0
    diagnostic_message: str = "Nominal"


class PATTracker:
    """
    Main tracking and state-estimation coordinator for FSOC coarse PAT.
    """

    def __init__(
        self,
        camera: VirtualCamera,
        prediction_horizon_steps: int = 5,
        q_process_spectral: float = 0.5,
        r_measurement_std_rad: float = 0.002,
        search_type: str = "spiral",
        use_hybrid_tracker: bool = False,
        enable_10_state: bool = False,
    ) -> None:
        self.camera = camera
        self.horizon_steps = prediction_horizon_steps
        self.use_hybrid = use_hybrid_tracker

        # Kalman Filter for target tracking in absolute world angular coordinates
        self.kalman = KalmanTracker(
            q_process_spectral=q_process_spectral,
            r_measurement_std=r_measurement_std_rad,
            gate_threshold_sigma=4.5,
        )

        # Adaptive Hybrid Kalman-Particle Filter
        self.hybrid = HybridPATTracker(
            q_process_spectral=q_process_spectral,
            r_measurement_std_rad=r_measurement_std_rad,
        )

        # Aerospace State Machine
        self.fsm = TrackingStateMachine(
            acquire_confirm_frames=3,
            uncertain_timeout_frames=12,
            lost_timeout_frames=25,
            reacquire_max_frames=60,
            enable_10_state=enable_10_state,
        )

        # Dynamic Safe FOV & Loss-of-Lock Risk Engine
        self.safe_fov_engine = DynamicSafeFOVEngine(camera=camera)

        # Fine-PAT Handoff Readiness Engine
        self.handoff_engine = FinePATHandoffEngine()

        # Search strategies
        self.search_type = search_type.lower()
        if self.search_type == "raster":
            self.search_strategy: SearchStrategy = RasterScan()
        else:
            self.search_strategy = SpiralScan()

        self.predictive_search = PredictiveSearch()

    def reset(self) -> None:
        """Reset tracker, Kalman filter, hybrid filter, and state machine."""
        self.kalman.reset()
        self.hybrid.reset()
        self.fsm.reset(initial_state=PATState.SEARCHING)
        self.handoff_engine.reset()
        self.search_strategy.reset(self.camera.pan, self.camera.tilt)

    def process_frame(
        self,
        detection: Optional[Detection],
        sim_time: float,
        dt: float,
        turbulence_level: float = 0.0,
        is_occluded: bool = False,
        vibration_amp_rad: float = 0.0,
    ) -> TrackerOutput:
        """
        Process incoming detection (or absence of detection) for current frame.
        """
        # Step 1: Predict Kalman state
        self.kalman.predict(dt)

        has_detection = detection is not None
        measured_world_pos: Optional[Tuple[float, float]] = None
        current_delta_az = 0.0
        current_delta_el = 0.0
        conf = detection.confidence if detection is not None else 0.0

        if has_detection:
            # Convert sensor pixel detection (u, v) into camera boresight relative angular error
            d_az, d_el = self.camera.projection.pixels_to_angular_error(
                detection.center_u, detection.center_v, exact=True
            )
            current_delta_az = d_az
            current_delta_el = d_el

            # Absolute target angular coordinate in world space
            target_az = self.camera.effective_pan + d_az
            target_el = self.camera.effective_tilt + d_el
            measured_world_pos = (target_az, target_el)

            # Update Kalman filter
            self.kalman.update((target_az, target_el), use_gating=True)

        # Update hybrid estimator
        hybrid_est = self.hybrid.step(
            dt=dt,
            meas_az=measured_world_pos[0] if measured_world_pos else None,
            meas_el=measured_world_pos[1] if measured_world_pos else None,
            confidence=conf,
            turbulence_level=turbulence_level,
            is_occluded=is_occluded,
            consecutive_misses=self.fsm.consecutive_misses,
        )

        tracking_err_rad = math.hypot(current_delta_az, current_delta_el) if has_detection else 0.005
        tracking_err_deg = rad2deg(tracking_err_rad)
        tracking_err_mrad = tracking_err_rad * 1e3

        # Step 2: State machine update
        is_centered = bool(has_detection and (tracking_err_rad < 0.003))
        new_state, state_changed = self.fsm.update(
            has_detection=has_detection,
            is_centered=is_centered,
            sim_time=sim_time,
            confidence=conf if has_detection else None,
            tracking_error_deg=tracking_err_deg if has_detection else None,
            is_in_fov=has_detection,
        )

        # If state just switched into SEARCHING or REACQUIRING, reset scan patterns
        if state_changed:
            if new_state == PATState.SEARCHING:
                self.search_strategy.reset(self.camera.pan, self.camera.tilt)
            elif new_state == PATState.REACQUIRING:
                pos_est = self.kalman.position
                vel_est = self.kalman.velocity
                self.predictive_search.set_prediction(pos_est, vel_est)

        cmd_delta_az = 0.0
        cmd_delta_el = 0.0
        search_rate: Optional[Tuple[float, float]] = None
        pred_world_pos: Optional[Tuple[float, float]] = None
        pred_pixel_uv: Optional[Tuple[float, float]] = None

        # Step 3: Compute actions according to state
        if new_state in (PATState.TRACKING, PATState.LOCKED, PATState.ACQUIRING):
            # Lead-ahead predictive control
            if self.horizon_steps > 0 and self.kalman.initialized:
                p_az, p_el, _, _ = self.kalman.predict_ahead(self.horizon_steps, dt)
                pred_world_pos = (p_az, p_el)

                cmd_delta_az = p_az - self.camera.effective_pan
                cmd_delta_el = p_el - self.camera.effective_tilt

                _, _, pu, pv, _ = self.camera.project_target(p_az, p_el)
                pred_pixel_uv = (pu, pv)
            else:
                cmd_delta_az = current_delta_az
                cmd_delta_el = current_delta_el

        elif new_state in (PATState.UNCERTAIN, PATState.DEGRADED):
            # Coasting on dead-reckoning during momentary signal loss
            p_az, p_el, _, _ = self.kalman.predict_ahead(self.horizon_steps, dt)
            pred_world_pos = (p_az, p_el)
            cmd_delta_az = p_az - self.camera.effective_pan
            cmd_delta_el = p_el - self.camera.effective_tilt

            _, _, pu, pv, _ = self.camera.project_target(p_az, p_el)
            pred_pixel_uv = (pu, pv)

        elif new_state == PATState.REACQUIRING:
            # Run localized adaptive predictive search
            rate_pan, rate_tilt = self.predictive_search.step(
                self.camera.pan, self.camera.tilt, dt
            )
            search_rate = (rate_pan, rate_tilt)

        elif new_state == PATState.SEARCHING:
            # Systematic wide-area scanning
            rate_pan, rate_tilt = self.search_strategy.step(
                self.camera.pan, self.camera.tilt, dt
            )
            search_rate = (rate_pan, rate_tilt)

        is_locked = (new_state == PATState.LOCKED) or (new_state == PATState.TRACKING and is_centered)

        # Target velocity estimate
        target_vel = self.kalman.velocity

        # Step 4: Dynamic Safe FOV & Loss-of-Lock Risk
        safe_fov_telem = self.safe_fov_engine.compute(
            rel_az_rad=current_delta_az,
            rel_el_rad=current_delta_el,
            target_vel_rad_s=target_vel,
            vibration_amp_rad=vibration_amp_rad,
            is_occluded=is_occluded,
            tracking_error_mrad=tracking_err_mrad,
            confidence=conf,
        )

        # Step 5: Fine-PAT Handoff Readiness
        handoff_telem = self.handoff_engine.update(
            tracking_error_mrad=tracking_err_mrad,
            is_locked_or_tracking=(new_state in (PATState.TRACKING, PATState.LOCKED)),
            target_vel_rad_s=target_vel,
            risk_state=safe_fov_telem.risk_state,
        )

        active_flt = hybrid_est.active_filter if self.use_hybrid else "KF"

        return TrackerOutput(
            state=new_state,
            state_changed=state_changed,
            cmd_delta_az_rad=cmd_delta_az,
            cmd_delta_el_rad=cmd_delta_el,
            search_rate_rad_s=search_rate,
            estimated_vel_rad_s=target_vel,
            predicted_world_pos_rad=pred_world_pos,
            predicted_pixel_uv=pred_pixel_uv,
            is_locked=is_locked,
            active_filter=active_flt,
            severity_score=hybrid_est.severity_score,
            handoff_score_pct=handoff_telem.handoff_score_pct,
            handoff_state=handoff_telem.handoff_state,
            risk_state=safe_fov_telem.risk_state,
            risk_score=safe_fov_telem.risk_score,
            safe_fov_box_pixels=safe_fov_telem.hud_box_pixels,
            dist_to_safe_edge_mrad=safe_fov_telem.dist_to_safe_edge_mrad,
            diagnostic_message=safe_fov_telem.diagnostic_message,
        )
