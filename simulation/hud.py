"""
HUD (Heads-Up Display) visual overlay renderer for FSOC camera tracking.

Renders:
- Camera Boresight Optical Center Crosshair
- Lock Dead-Zone Box/Circle
- Detected Target Bounding Box and Center
- Ground Truth Target Marker (for benchmarking & validation)
- Real-time Flight/Optical Telemetry HUD Overlay
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from vision.detection_types import Detection
from simulation.coordinate_system import rad2deg


def _draw_hud_text(
    canvas: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.48,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    with_shadow: bool = True,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Render anti-aliased HUD text with optional high-contrast drop shadow."""
    x, y = pos
    if with_shadow:
        cv2.putText(
            canvas,
            text,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            shadow_color,
            thickness + 1,
            lineType=cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_badge_pill(
    canvas: np.ndarray,
    text: str,
    top_left: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    border_color: Optional[Tuple[int, int, int]] = None,
    font_scale: float = 0.45,
    padding: Tuple[int, int] = (8, 4),
) -> Tuple[int, int, int, int]:
    """Render a crisp status badge pill with background, optional border, and centered text."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad_x, pad_y = padding
    x, y = top_left
    box_w = tw + pad_x * 2
    box_h = th + pad_y * 2
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), bg_color, -1)
    if border_color is not None:
        cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), border_color, 1, lineType=cv2.LINE_AA)
    text_x = x + pad_x
    text_y = y + pad_y + th
    _draw_hud_text(
        canvas,
        text,
        (text_x, text_y),
        font_scale=font_scale,
        color=text_color,
        thickness=1,
        with_shadow=True,
    )
    return (x, y, box_w, box_h)


def render_hud_overlay(
    frame: np.ndarray,
    detection: Optional[Detection],
    ground_truth_uv: Optional[Tuple[float, float]],
    camera_pan_deg: float,
    camera_tilt_deg: float,
    angular_error_deg: float,
    dead_zone_pixels: float = 8.0,
    fps: float = 30.0,
    state_str: str = "TRACKING",
    is_locked: bool = False,
    predicted_uv: Optional[Tuple[float, float]] = None,
    safe_fov_box_pixels: Optional[Tuple[int, int, int, int]] = None,
    risk_state: str = "STABLE",
    handoff_score_pct: float = 0.0,
    handoff_state: str = "NOT READY",
    active_filter: str = "KF",
) -> np.ndarray:
    """
    Apply professional mission-control HUD graphics onto camera viewport frame.
    All text rendering utilizes anti-aliased rasterization (LINE_AA), drop-shadow
    geometry, and solid status badge pills for razor-sharp legibility.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    cx, cy = int(w / 2), int(h / 2)

    # 1. Camera Center Boresight Reticle
    reticle_color = (255, 220, 0)  # Cyan in BGR
    cv2.line(canvas, (cx - 24, cy), (cx - 7, cy), reticle_color, 1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (cx + 7, cy), (cx + 24, cy), reticle_color, 1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 24), (cx, cy - 7), reticle_color, 1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + 7), (cx, cy + 24), reticle_color, 1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 2, reticle_color, -1, lineType=cv2.LINE_AA)

    # 2. Dead-zone boundary
    dz_r = int(round(dead_zone_pixels))
    dz_color = (0, 255, 120) if is_locked else (80, 100, 120)
    cv2.circle(canvas, (cx, cy), dz_r, dz_color, 1, lineType=cv2.LINE_AA)

    # 2b. Dynamic Safe FOV Boundary Box (Aerospace Section 16.2)
    if safe_fov_box_pixels is not None:
        sx1, sy1, sx2, sy2 = safe_fov_box_pixels
        sfov_col = (255, 200, 0) if risk_state == "STABLE" else ((0, 160, 255) if risk_state == "WARNING" else (60, 60, 255))
        # Draw subtle bounding box with corner brackets
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), sfov_col, 1, lineType=cv2.LINE_AA)
        c_len = 12
        cv2.line(canvas, (sx1, sy1), (sx1 + c_len, sy1), sfov_col, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (sx1, sy1), (sx1, sy1 + c_len), sfov_col, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (sx2, sy2), (sx2 - c_len, sy2), sfov_col, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (sx2, sy2), (sx2, sy2 - c_len), sfov_col, 2, lineType=cv2.LINE_AA)
        _draw_badge_pill(
            canvas,
            "SAFE FOV",
            (sx1 + 4, max(46, sy1 + 4)),
            bg_color=(15, 20, 30),
            text_color=sfov_col,
            border_color=sfov_col,
            font_scale=0.38,
            padding=(5, 3),
        )

    # 3. Ground Truth Marker (Gold / Yellow Dashed Ring)
    if ground_truth_uv is not None:
        gt_x, gt_y = int(round(ground_truth_uv[0])), int(round(ground_truth_uv[1]))
        if 0 <= gt_x < w and 0 <= gt_y < h:
            cv2.circle(canvas, (gt_x, gt_y), 13, (0, 215, 255), 1, lineType=cv2.LINE_AA)
            cv2.drawMarker(
                canvas,
                (gt_x, gt_y),
                (0, 215, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=8,
                thickness=1,
            )
            _draw_badge_pill(
                canvas,
                "GT",
                (gt_x + 14, max(46, gt_y - 10)),
                bg_color=(15, 20, 28),
                text_color=(0, 225, 255),
                border_color=(0, 200, 240),
                font_scale=0.38,
                padding=(5, 2),
            )

    # 4. Predicted Position Marker (Magenta)
    if predicted_uv is not None:
        p_x, p_y = int(round(predicted_uv[0])), int(round(predicted_uv[1]))
        if 0 <= p_x < w and 0 <= p_y < h:
            cv2.drawMarker(
                canvas,
                (p_x, p_y),
                (255, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
            )
            _draw_badge_pill(
                canvas,
                "PRED",
                (p_x + 12, max(46, p_y - 10)),
                bg_color=(25, 12, 28),
                text_color=(255, 100, 255),
                border_color=(220, 0, 220),
                font_scale=0.38,
                padding=(5, 2),
            )

    # 5. Detected Target Bounding Box & Centroid (Vibrant Green)
    if detection is not None:
        bx = detection.bbox
        det_color = (0, 255, 100)
        cv2.rectangle(
            canvas,
            (bx.x, bx.y),
            (bx.x + bx.w, bx.y + bx.h),
            det_color,
            1,
            lineType=cv2.LINE_AA,
        )

        # Corner brackets
        corner_len = min(8, bx.w // 2, bx.h // 2)
        cv2.line(canvas, (bx.x, bx.y), (bx.x + corner_len, bx.y), det_color, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (bx.x, bx.y), (bx.x, bx.y + corner_len), det_color, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (bx.x + bx.w, bx.y + bx.h), (bx.x + bx.w - corner_len, bx.y + bx.h), det_color, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, (bx.x + bx.w, bx.y + bx.h), (bx.x + bx.w, bx.y + bx.h - corner_len), det_color, 2, lineType=cv2.LINE_AA)

        det_cx, det_cy = int(round(detection.center_u)), int(round(detection.center_v))
        cv2.circle(canvas, (det_cx, det_cy), 3, (0, 255, 100), -1, lineType=cv2.LINE_AA)

        # Error vector line from center to target
        cv2.line(canvas, (cx, cy), (det_cx, det_cy), (0, 190, 255), 1, lineType=cv2.LINE_AA)

        label = f"BEACON {detection.confidence * 100:.0f}%"
        badge_y = max(46, bx.y - 20)
        _draw_badge_pill(
            canvas,
            label,
            (bx.x, badge_y),
            bg_color=(12, 28, 16),
            text_color=(0, 255, 120),
            border_color=(0, 200, 80),
            font_scale=0.40,
            padding=(6, 3),
        )

    # 6. Status Banners & Telemetry Top/Bottom Bars (High Resolution & Contrast)
    top_bar_h = 44
    bottom_bar_h = 38
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (12, 16, 24), -1)
    cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    # Accent divider lines
    cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (35, 48, 68), 1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (0, h - bottom_bar_h), (w, h - bottom_bar_h), (35, 48, 68), 1, lineType=cv2.LINE_AA)

    # State badge (Left of Top Bar)
    state_bg_color = (0, 130, 45) if state_str in ("TRACKING", "LOCKED") else (0, 95, 200)
    state_border = (0, 230, 90) if state_str in ("TRACKING", "LOCKED") else (0, 170, 255)
    if state_str == "LOST":
        state_bg_color = (25, 25, 175)
        state_border = (70, 70, 255)
    elif state_str in ("SEARCHING", "REACQUIRING"):
        state_bg_color = (160, 100, 0)
        state_border = (255, 180, 0)
    elif state_str == "DEGRADED":
        state_bg_color = (0, 75, 180)
        state_border = (0, 140, 240)

    _draw_badge_pill(
        canvas,
        f"PAT: {state_str}",
        (10, 9),
        bg_color=state_bg_color,
        text_color=(255, 255, 255),
        border_color=state_border,
        font_scale=0.48,
        padding=(10, 5),
    )

    # Telemetry text (Crisp, High Contrast, Anti-Aliased)
    telem_str = (
        f"PAN: {camera_pan_deg:+05.1f}deg  |  TILT: {camera_tilt_deg:+05.1f}deg  |  "
        f"ERR: {angular_error_deg:05.3f}deg  |  FLT: {active_filter}  |  "
        f"FPS: {fps:4.1f}"
    )
    _draw_hud_text(
        canvas,
        telem_str,
        (170, 28),
        font_scale=0.48,
        color=(255, 235, 80),  # Bright electric cyan in BGR
        thickness=1,
        with_shadow=True,
    )

    # Risk state badge on top-right
    risk_col = (0, 220, 90) if risk_state == "STABLE" else ((0, 185, 255) if risk_state == "WARNING" else (60, 60, 255))
    risk_text = f"RISK: {risk_state}"
    (rt_w, _), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    risk_x = max(w - rt_w - 28, w - 170)
    _draw_badge_pill(
        canvas,
        risk_text,
        (risk_x, 9),
        bg_color=(20, 24, 34),
        text_color=risk_col,
        border_color=risk_col,
        font_scale=0.46,
        padding=(8, 5),
    )

    # --- Bottom Bar: High-Visibility Fine-PAT Handoff Status & Progress Indicator ---
    if handoff_state == "READY":
        hf_bg = (16, 50, 24)
        hf_border = (0, 240, 100)
        hf_text_col = (0, 255, 120)
    elif handoff_state == "STABILIZING":
        hf_bg = (18, 42, 54)
        hf_border = (0, 190, 255)
        hf_text_col = (0, 225, 255)
    else:  # NOT READY
        hf_bg = (24, 20, 36)
        hf_border = (70, 80, 220)
        hf_text_col = (90, 110, 255)  # High-contrast vibrant coral-red, never dim gray!

    handoff_text = f"FINE-PAT HANDOFF: {handoff_score_pct:.0f}% [{handoff_state}]"
    _draw_badge_pill(
        canvas,
        handoff_text,
        (10, h - bottom_bar_h + 6),
        bg_color=hf_bg,
        text_color=hf_text_col,
        border_color=hf_border,
        font_scale=0.48,
        padding=(10, 4),
    )

    # Mini Progress Bar for Fine-PAT Readiness Score
    bar_x = 350
    bar_y = h - bottom_bar_h + 12
    bar_w = 70
    bar_h = 14
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 30, 42), -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 75, 100), 1, lineType=cv2.LINE_AA)
    fill_w = int(np.clip(round((handoff_score_pct / 100.0) * bar_w), 0, bar_w))
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x + 1, bar_y + 1), (bar_x + fill_w - 1, bar_y + bar_h - 1), hf_border, -1)

    # Legend on bottom-right (Anti-Aliased, High Legibility)
    legend_str = f"[GT: Gold] [Beacon: Green] [SafeFOV: Cyan] [Filter: {active_filter}]"
    _draw_hud_text(
        canvas,
        legend_str,
        (w - 480, h - 14),
        font_scale=0.42,
        color=(215, 225, 235),
        thickness=1,
        with_shadow=True,
    )

    return canvas
