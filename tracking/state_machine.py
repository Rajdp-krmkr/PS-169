"""
PAT (Pointing, Acquisition, and Tracking) Finite State Machine.

Aerospace 10-State Model (ISRO / PS-26169 Compliant):
- IDLE            : Standby mode before simulation initiation.
- SEARCHING       : Systematic spiral/raster search pattern across Field of Regard.
- CANDIDATE_FOUND : Initial optical candidate detected with low confidence (< 0.48).
- VERIFYING       : Candidate persistence verification across M-of-N confirmation window.
- ACQUIRING       : Confirmed signal; actively slewing camera toward beacon centroid.
- TRACKING        : Closed-loop continuous coarse tracking.
- LOCKED          : High-precision lock established (pointing error < 1.0 mrad).
- DEGRADED        : Optical signal degraded by scintillation/turbulence while target remains in FOV.
- UNCERTAIN       : Temporary detection loss; coasting on Kalman/Particle filter prediction.
- LOST            : Target lost outside FOV or occluded past timeout threshold.
- REACQUIRING     : Localized predictive search around propagated covariance ellipse.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import time


class PATState(Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    CANDIDATE_FOUND = "CANDIDATE_FOUND"
    VERIFYING = "VERIFYING"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    LOCKED = "LOCKED"
    DEGRADED = "DEGRADED"
    UNCERTAIN = "UNCERTAIN"  # Preserved for backward compatibility
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


class TrackingStateMachine:
    """
    Robust state machine with hysteresis, confirmation counters, timeout monitoring,
    and aerospace 10-state acquisition lifecycle.
    """

    def __init__(
        self,
        acquire_confirm_frames: int = 3,
        uncertain_timeout_frames: int = 12,
        lost_timeout_frames: int = 30,
        reacquire_max_frames: int = 60,
        enable_10_state: bool = False,
        lock_error_threshold_deg: float = 0.0573,  # ~1.0 mrad
    ) -> None:
        self.acquire_confirm_needed = acquire_confirm_frames
        self.uncertain_timeout = uncertain_timeout_frames
        self.lost_timeout = lost_timeout_frames
        self.reacquire_max = reacquire_max_frames
        self.enable_10_state = enable_10_state
        self.lock_error_thresh_deg = lock_error_threshold_deg

        self.current_state: PATState = PATState.SEARCHING
        self.previous_state: PATState = PATState.SEARCHING

        # Counters
        self.consecutive_detections: int = 0
        self.consecutive_misses: int = 0
        self.reacquire_frames: int = 0

        # State timing telemetry
        self.time_in_state_frames: int = 0
        self.total_lost_events: int = 0
        self.total_reacquisitions: int = 0

        # Timestamp tracking for reacquisition latency
        self.loss_start_sim_time: Optional[float] = None
        self.last_reacquisition_latency: float = 0.0

    def reset(self, initial_state: PATState = PATState.SEARCHING) -> None:
        """Reset state machine counters and state."""
        self.current_state = initial_state
        self.previous_state = initial_state
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.reacquire_frames = 0
        self.time_in_state_frames = 0
        self.loss_start_sim_time = None

    def update(
        self,
        has_detection: bool,
        is_centered: bool = False,
        sim_time: float = 0.0,
        confidence: Optional[float] = None,
        tracking_error_deg: Optional[float] = None,
        is_in_fov: bool = True,
    ) -> Tuple[PATState, bool]:
        """
        Advance state machine given current frame observation.
        Returns: (new_state, state_changed_bool)
        """
        state_changed = False
        prev = self.current_state

        if has_detection:
            self.consecutive_detections += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            self.consecutive_detections = 0

        # Granular confidence gating if provided
        conf = confidence if confidence is not None else (1.0 if has_detection else 0.0)

        # -----------------------------------------------------------------
        # Transition Logic
        # -----------------------------------------------------------------
        if self.current_state in (PATState.IDLE, PATState.SEARCHING):
            if has_detection:
                if self.enable_10_state and conf < 0.48:
                    self.current_state = PATState.CANDIDATE_FOUND
                elif self.enable_10_state and conf < 0.65:
                    self.current_state = PATState.VERIFYING
                else:
                    self.current_state = PATState.ACQUIRING
                self.time_in_state_frames = 0

        elif self.current_state == PATState.CANDIDATE_FOUND:
            if not has_detection:
                self.current_state = PATState.SEARCHING
            elif conf >= 0.48:
                self.current_state = PATState.VERIFYING

        elif self.current_state == PATState.VERIFYING:
            if not has_detection:
                self.current_state = PATState.SEARCHING
            elif self.consecutive_detections >= 2:
                self.current_state = PATState.ACQUIRING

        elif self.current_state == PATState.ACQUIRING:
            if not has_detection:
                self.current_state = PATState.SEARCHING
                self.time_in_state_frames = 0
            elif self.consecutive_detections >= self.acquire_confirm_needed:
                # Transition to TRACKING or LOCKED based on error
                if self.enable_10_state and tracking_error_deg is not None and tracking_error_deg <= self.lock_error_thresh_deg:
                    self.current_state = PATState.LOCKED
                else:
                    self.current_state = PATState.TRACKING
                self.time_in_state_frames = 0

        elif self.current_state in (PATState.TRACKING, PATState.LOCKED):
            if not has_detection:
                if self.enable_10_state and is_in_fov:
                    self.current_state = PATState.DEGRADED
                else:
                    self.current_state = PATState.UNCERTAIN
                self.time_in_state_frames = 0
                self.loss_start_sim_time = sim_time
            else:
                # High-precision lock promotion / demotion
                if self.enable_10_state and tracking_error_deg is not None:
                    if tracking_error_deg <= self.lock_error_thresh_deg:
                        self.current_state = PATState.LOCKED
                    else:
                        self.current_state = PATState.TRACKING

        elif self.current_state in (PATState.UNCERTAIN, PATState.DEGRADED):
            if has_detection:
                # Recovered immediately from momentary occlusion / turbulence fade
                if self.enable_10_state and tracking_error_deg is not None and tracking_error_deg <= self.lock_error_thresh_deg:
                    self.current_state = PATState.LOCKED
                else:
                    self.current_state = PATState.TRACKING
                self.time_in_state_frames = 0
                self.loss_start_sim_time = None
            elif self.consecutive_misses > self.uncertain_timeout:
                self.current_state = PATState.LOST
                self.total_lost_events += 1
                self.time_in_state_frames = 0

        elif self.current_state == PATState.LOST:
            if has_detection:
                self.current_state = PATState.ACQUIRING
                self.time_in_state_frames = 0
                if self.loss_start_sim_time is not None:
                    self.last_reacquisition_latency = sim_time - self.loss_start_sim_time
                    self.total_reacquisitions += 1
                    self.loss_start_sim_time = None
            else:
                self.current_state = PATState.REACQUIRING
                self.reacquire_frames = 0
                self.time_in_state_frames = 0

        elif self.current_state == PATState.REACQUIRING:
            self.reacquire_frames += 1
            if has_detection:
                self.current_state = PATState.ACQUIRING
                self.time_in_state_frames = 0
                if self.loss_start_sim_time is not None:
                    self.last_reacquisition_latency = sim_time - self.loss_start_sim_time
                    self.total_reacquisitions += 1
                    self.loss_start_sim_time = None
            elif self.reacquire_frames > self.reacquire_max:
                # Fallback to wide-area systematic search
                self.current_state = PATState.SEARCHING
                self.time_in_state_frames = 0

        if self.current_state != prev:
            state_changed = True
            self.previous_state = prev
            self.time_in_state_frames = 0
        else:
            self.time_in_state_frames += 1

        return self.current_state, state_changed
