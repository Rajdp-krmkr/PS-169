"""
Dynamic Safe FOV and Loss-of-Lock Risk Engine for FSOC Coarse PAT.

Calculates:
- Real-time Safe FOV boundaries inside the coarse optical sensor based on
  target angular velocity vector and platform jitter/vibration amplitude.
- Milliradian angular distance to safe boundary.
- Real-time Loss-of-Lock Risk Score (0.0 to 1.0) and Risk State:
  - STABLE: Beacon securely tracking well within safe envelope.
  - WARNING: Beacon approaching safe FOV boundary.
  - LOCK AT RISK: Critical proximity to boundary or high pointing error rate.
  - CRITICAL: Boundary excursion or severe Line-of-Sight occlusion.
- Root-cause natural language diagnostic explanations.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from simulation.coordinate_system import rad2deg, deg2rad
from simulation.camera import VirtualCamera


@dataclass
class SafeFOVTelemetry:
    # Angular bounds in camera-relative coordinates (radians)
    safe_az_min_rad: float
    safe_az_max_rad: float
    safe_el_min_rad: float
    safe_el_max_rad: float
    # Safe dimensions in milliradians
    safe_span_az_mrad: float
    safe_span_el_mrad: float
    # Distance to safe edge in mrad (negative means outside safe boundary)
    dist_to_safe_edge_mrad: float
    # Risk assessment
    risk_score: float  # 0.0 to 1.0
    risk_state: str    # "STABLE", "WARNING", "LOCK AT RISK", "CRITICAL"
    # Root-cause diagnostic explanation
    diagnostic_message: str
    # Pixel box for HUD rendering (xmin, ymin, xmax, ymax)
    hud_box_pixels: Optional[Tuple[int, int, int, int]] = None


class DynamicSafeFOVEngine:
    """
    Computes dynamic safe tracking margins and predicts loss-of-lock risk before
    target exits camera field of view.
    """

    def __init__(
        self,
        camera: VirtualCamera,
        min_margin_pct: float = 0.08,
        max_margin_pct: float = 0.35,
    ) -> None:
        self.camera = camera
        self.min_margin_pct = min_margin_pct
        self.max_margin_pct = max_margin_pct

    def compute(
        self,
        rel_az_rad: float,
        rel_el_rad: float,
        target_vel_rad_s: Tuple[float, float] = (0.0, 0.0),
        vibration_amp_rad: float = 0.0,
        is_occluded: bool = False,
        tracking_error_mrad: Optional[float] = None,
        confidence: float = 1.0,
    ) -> SafeFOVTelemetry:
        """
        Evaluate dynamic safe boundary and loss-of-lock risk.

        Args:
            rel_az_rad: Target azimuth relative to camera boresight (rad)
            rel_el_rad: Target elevation relative to camera boresight (rad)
            target_vel_rad_s: Target angular velocity vector (rad/s)
            vibration_amp_rad: Current platform vibration amplitude (rad)
            is_occluded: Whether optical line-of-sight is physically occluded
            tracking_error_mrad: Current radial tracking error in mrad
            confidence: Detector confidence score [0.0, 1.0]
        """
        fov_h = getattr(self.camera, "fov_h_rad", getattr(self.camera, "fov_horizontal_rad", 0.35))
        fov_v = getattr(self.camera, "fov_v_rad", getattr(self.camera, "fov_vertical_rad", 0.20))
        half_fov_az = fov_h / 2.0
        half_fov_el = fov_v / 2.0

        # Target angular speed norm in mrad/s
        speed_norm_mrad_s = math.hypot(target_vel_rad_s[0], target_vel_rad_s[1]) * 1e3
        vib_amp_mrad = vibration_amp_rad * 1e3

        # Dynamic margins scaled with target kinematics and platform jitter
        # Margin increases when target is fast or platform is shaking heavily
        base_az_margin = half_fov_az * self.min_margin_pct
        base_el_margin = half_fov_el * self.min_margin_pct
        max_az_margin = half_fov_az * self.max_margin_pct
        max_el_margin = half_fov_el * self.max_margin_pct

        dynamic_az_add = (speed_norm_mrad_s * 1e-3 * 1.2) + (vib_amp_mrad * 1e-3 * 3.5)
        dynamic_el_add = (speed_norm_mrad_s * 1e-3 * 1.0) + (vib_amp_mrad * 1e-3 * 3.0)

        margin_az = min(max_az_margin, max(base_az_margin, base_az_margin + dynamic_az_add))
        margin_el = min(max_el_margin, max(base_el_margin, base_el_margin + dynamic_el_add))

        safe_az_max = half_fov_az - margin_az
        safe_az_min = -safe_az_max
        safe_el_max = half_fov_el - margin_el
        safe_el_min = -safe_el_max

        safe_span_az_mrad = (safe_az_max - safe_az_min) * 1e3
        safe_span_el_mrad = (safe_el_max - safe_el_min) * 1e3

        # Distance from target to safe boundary in mrad
        dist_edge_az_mrad = (safe_az_max - abs(rel_az_rad)) * 1e3
        dist_edge_el_mrad = (safe_el_max - abs(rel_el_rad)) * 1e3
        dist_to_safe_edge_mrad = min(dist_edge_az_mrad, dist_edge_el_mrad)

        if tracking_error_mrad is None:
            tracking_error_mrad = math.hypot(rel_az_rad, rel_el_rad) * 1e3

        # Loss-of-Lock Risk Score & State Categorization
        if is_occluded:
            risk_state = "CRITICAL"
            risk_score = 0.98
            diag = "Critical: Line-of-sight occlusion detected; executing predictive kinematic coast."
        elif dist_to_safe_edge_mrad < -2.0 or abs(rel_az_rad) > half_fov_az or abs(rel_el_rad) > half_fov_el:
            risk_state = "CRITICAL"
            risk_score = 0.95
            diag = f"Critical pointing excursion (Error: {tracking_error_mrad:.2f} mrad) exceeding coarse FOV safety envelope."
        elif dist_to_safe_edge_mrad < 1.0 or tracking_error_mrad > 4.0:
            risk_state = "LOCK AT RISK"
            risk_score = 0.75
            diag = f"Lock at risk: Target approaching Safe FOV boundary ({dist_to_safe_edge_mrad:.1f} mrad margin) with elevated slew."
        elif dist_to_safe_edge_mrad < 4.0 or tracking_error_mrad > 2.0 or confidence < 0.45:
            risk_state = "WARNING"
            risk_score = 0.48
            diag = f"Warning: Safe boundary proximity ({dist_to_safe_edge_mrad:.1f} mrad margin) or degraded optical confidence ({confidence:.2f})."
        else:
            risk_state = "STABLE"
            risk_score = max(0.05, min(0.30, 0.10 + (tracking_error_mrad / 15.0)))
            diag = f"Nominal closed-loop PAT tracking. Pointing error within threshold ({tracking_error_mrad:.3f} mrad)."

        # Pixel coordinates of safe box on image plane
        w, h = self.camera.resolution
        u_min = int(round((cx := w / 2.0) + (safe_az_min / fov_h) * w))
        u_max = int(round(cx + (safe_az_max / fov_h) * w))
        v_min = int(round((cy := h / 2.0) - (safe_el_max / fov_v) * h))
        v_max = int(round(cy - (safe_el_min / fov_v) * h))
        hud_box = (max(0, u_min), max(0, v_min), min(w - 1, u_max), min(h - 1, v_max))

        return SafeFOVTelemetry(
            safe_az_min_rad=safe_az_min,
            safe_az_max_rad=safe_az_max,
            safe_el_min_rad=safe_el_min,
            safe_el_max_rad=safe_el_max,
            safe_span_az_mrad=round(safe_span_az_mrad, 1),
            safe_span_el_mrad=round(safe_span_el_mrad, 1),
            dist_to_safe_edge_mrad=round(dist_to_safe_edge_mrad, 2),
            risk_score=round(risk_score, 2),
            risk_state=risk_state,
            diagnostic_message=diag,
            hud_box_pixels=hud_box,
        )
