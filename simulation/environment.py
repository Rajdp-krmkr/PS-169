"""
Virtual 2D celestial/angular world environment for FSOC PAT simulation.

Manages:
- Angular world boundaries (Azimuth & Elevation)
- Primary optical beacon and optical distractors/clutter
- Persistent starfield with world-fixed stars projected into camera FOV
- Global panoramic overview visualization for mission-control display
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from simulation.coordinate_system import deg2rad, rad2deg
from simulation.beacon import Beacon
from simulation.camera import VirtualCamera
from simulation.trajectory import create_trajectory


class VirtualEnvironment:
    """
    Manages the simulated angular celestial world, targets, and background stars.
    """

    def __init__(
        self,
        width_deg: float = 60.0,
        height_deg: float = 35.0,
        num_stars: int = 150,
        ambient_light: int = 10,
        seed: int = 42,
    ) -> None:
        self.width_rad = deg2rad(width_deg)
        self.height_rad = deg2rad(height_deg)
        self.az_min = -self.width_rad / 2.0
        self.az_max = self.width_rad / 2.0
        self.el_min = -self.height_rad / 2.0
        self.el_max = self.height_rad / 2.0

        self.ambient_light = ambient_light
        self.beacons: List[Beacon] = []

        # Generate world-fixed stars [az_rad, el_rad, brightness]
        rng = np.random.default_rng(seed)
        star_az = rng.uniform(self.az_min, self.az_max, num_stars)
        star_el = rng.uniform(self.el_min, self.el_max, num_stars)
        star_bri = rng.integers(25, 110, num_stars, dtype=np.uint8)
        self.stars = list(zip(star_az, star_el, star_bri))

    def add_beacon(self, beacon: Beacon) -> None:
        """Add a target or distractor to the environment."""
        self.beacons.append(beacon)

    def get_primary_beacon(self) -> Optional[Beacon]:
        """Retrieve the primary designated beacon."""
        for b in self.beacons:
            if b.is_primary:
                return b
        return self.beacons[0] if self.beacons else None

    def update(self, t: float, dt: float) -> None:
        """Update all active beacons."""
        for b in self.beacons:
            b.update(t, dt)

    def generate_camera_background(self, camera: VirtualCamera) -> np.ndarray:
        """
        Render world-fixed stars projected into the camera viewport.
        Stars drift accurately as the camera pans and tilts.
        """
        frame = np.full(
            (camera.height, camera.width, 3),
            self.ambient_light,
            dtype=np.uint8,
        )

        for s_az, s_el, bri in self.stars:
            delta_az, delta_el, u, v, in_fov = camera.project_target(s_az, s_el)
            if in_fov:
                ix, iy = int(round(u)), int(round(v))
                if 0 <= ix < camera.width and 0 <= iy < camera.height:
                    frame[iy, ix] = [bri, bri, bri]

        return frame

    def render_overview_map(
        self,
        camera: VirtualCamera,
        map_size: Tuple[int, int] = (480, 280),
    ) -> np.ndarray:
        """
        Render a global wide-angle tactical overview map showing the entire
        simulation world, all beacons, trajectories, and camera FOV footprint.
        All graphics and text utilize anti-aliasing (LINE_AA) and high-contrast
        tactical mission badges for crystal-clear legibility.
        """
        mw, mh = map_size
        map_img = np.full((mh, mw, 3), (16, 20, 28), dtype=np.uint8)

        # Coordinate transform from world radians to map pixels (accounting for header)
        header_h = 26
        content_y_min = header_h + 2
        content_y_max = mh - 6

        def world_to_map(az_rad: float, el_rad: float) -> Tuple[int, int]:
            norm_x = (az_rad - self.az_min) / (self.az_max - self.az_min)
            norm_y = (el_rad - self.el_min) / (self.el_max - self.el_min)
            mx = int(np.clip(norm_x * mw, 0, mw - 1))
            my = int(np.clip(content_y_max - norm_y * (content_y_max - content_y_min), content_y_min, content_y_max))
            return mx, my

        # Draw celestial grid lines
        for deg_x in range(-30, 31, 15):
            gx, _ = world_to_map(deg2rad(deg_x), 0)
            cv2.line(map_img, (gx, header_h), (gx, mh), (32, 40, 52), 1)
        for deg_y in range(-15, 16, 10):
            _, gy = world_to_map(0, deg2rad(deg_y))
            cv2.line(map_img, (0, gy), (mw, gy), (32, 40, 52), 1)

        # Draw background stars
        for s_az, s_el, _ in self.stars:
            sx, sy = world_to_map(s_az, s_el)
            if header_h < sy < mh:
                map_img[sy, sx] = [100, 110, 125]

        # Draw camera FOV bounding rectangle in world space
        c_pan = camera.effective_pan
        c_tilt = camera.effective_tilt
        half_fov_h = camera.fov_h_rad / 2.0
        half_fov_v = camera.fov_v_rad / 2.0

        p1 = world_to_map(c_pan - half_fov_h, c_tilt + half_fov_v)
        p2 = world_to_map(c_pan + half_fov_h, c_tilt - half_fov_v)
        cv2.rectangle(map_img, p1, p2, (255, 200, 0), 1, lineType=cv2.LINE_AA)

        # Draw camera boresight point
        cx_m, cy_m = world_to_map(c_pan, c_tilt)
        cv2.drawMarker(
            map_img,
            (cx_m, cy_m),
            (255, 220, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=9,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        # Draw beacons and history
        for b in self.beacons:
            # Trajectory history trail
            pts = [world_to_map(h_az, h_el) for h_az, h_el in b.history]
            if len(pts) > 1:
                trail_color = (0, 210, 90) if b.is_primary else (80, 80, 220)
                for i in range(len(pts) - 1):
                    cv2.line(map_img, pts[i], pts[i + 1], trail_color, 1, lineType=cv2.LINE_AA)

            bx, by = world_to_map(b.pos[0], b.pos[1])
            if b.is_primary:
                # Glowing beacon indicator
                cv2.circle(map_img, (bx, by), 5, (0, 255, 120), 1, lineType=cv2.LINE_AA)
                cv2.circle(map_img, (bx, by), 3, (0, 255, 120), -1, lineType=cv2.LINE_AA)

                # Crisp beacon pill badge
                b_x1 = min(bx + 7, mw - 62)
                b_y1 = max(header_h + 3, by - 14)
                cv2.rectangle(map_img, (b_x1, b_y1), (b_x1 + 54, b_y1 + 14), (14, 28, 18), -1)
                cv2.rectangle(map_img, (b_x1, b_y1), (b_x1 + 54, b_y1 + 14), (0, 200, 80), 1, lineType=cv2.LINE_AA)
                cv2.putText(
                    map_img,
                    "BEACON",
                    (b_x1 + 4, b_y1 + 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 120),
                    1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(map_img, (bx, by), 4, (0, 140, 255), 1, lineType=cv2.LINE_AA)
                cv2.circle(map_img, (bx, by), 2, (0, 140, 255), -1, lineType=cv2.LINE_AA)

        # Tactical Header Bar (Solid Aerospace Slate + Accent Line)
        cv2.rectangle(map_img, (0, 0), (mw, header_h), (12, 16, 26), -1)
        cv2.line(map_img, (0, header_h), (mw, header_h), (0, 180, 240), 1, lineType=cv2.LINE_AA)

        # Prominent "GLOBAL WORLD VIEW" Title (Drop shadow + LINE_AA)
        title_str = "GLOBAL WORLD VIEW"
        cv2.putText(
            map_img,
            title_str,
            (9, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            map_img,
            title_str,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 225, 0),  # Electric cyan in BGR
            1,
            lineType=cv2.LINE_AA,
        )

        # Sub-header telemetry badge on top-right of tactical radar
        sub_str = "RADAR: 60x35deg"
        (sub_w, _), _ = cv2.getTextSize(sub_str, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(
            map_img,
            sub_str,
            (mw - sub_w - 8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (175, 205, 230),
            1,
            lineType=cv2.LINE_AA,
        )

        # Outer Tactical Border & Glowing Corner Brackets
        cv2.rectangle(map_img, (0, 0), (mw - 1, mh - 1), (0, 150, 200), 1, lineType=cv2.LINE_AA)
        c_len = 8
        c_col = (0, 230, 255)
        # Top-left
        cv2.line(map_img, (0, 0), (c_len, 0), c_col, 2, lineType=cv2.LINE_AA)
        cv2.line(map_img, (0, 0), (0, c_len), c_col, 2, lineType=cv2.LINE_AA)
        # Top-right
        cv2.line(map_img, (mw - 1, 0), (mw - 1 - c_len, 0), c_col, 2, lineType=cv2.LINE_AA)
        cv2.line(map_img, (mw - 1, 0), (mw - 1, c_len), c_col, 2, lineType=cv2.LINE_AA)
        # Bottom-left
        cv2.line(map_img, (0, mh - 1), (c_len, mh - 1), c_col, 2, lineType=cv2.LINE_AA)
        cv2.line(map_img, (0, mh - 1), (0, mh - 1 - c_len), c_col, 2, lineType=cv2.LINE_AA)
        # Bottom-right
        cv2.line(map_img, (mw - 1, mh - 1), (mw - 1 - c_len, mh - 1), c_col, 2, lineType=cv2.LINE_AA)
        cv2.line(map_img, (mw - 1, mh - 1), (mw - 1, mh - 1 - c_len), c_col, 2, lineType=cv2.LINE_AA)

        return map_img


def build_environment_from_config(cfg: Dict[str, Any]) -> Tuple[VirtualEnvironment, VirtualCamera]:
    """Factory creating configured environment and camera."""
    w_cfg = cfg.get("world", {})
    env = VirtualEnvironment(
        width_deg=w_cfg.get("width_deg", 60.0),
        height_deg=w_cfg.get("height_deg", 35.0),
        ambient_light=w_cfg.get("ambient_light", 10),
    )

    c_cfg = cfg.get("camera", {})
    cam = VirtualCamera(
        resolution=tuple(c_cfg.get("resolution", [1280, 720])),
        fov_horizontal_deg=c_cfg.get("fov_horizontal_deg", 20.0),
        fov_vertical_deg=c_cfg.get("fov_vertical_deg", 11.25),
        pan_limits_deg=tuple(c_cfg.get("pan_limits_deg", [-60.0, 60.0])),
        tilt_limits_deg=tuple(c_cfg.get("tilt_limits_deg", [-35.0, 35.0])),
        max_pan_speed_deg_s=c_cfg.get("max_pan_speed_deg_s", 25.0),
        max_tilt_speed_deg_s=c_cfg.get("max_tilt_speed_deg_s", 25.0),
        fps=c_cfg.get("fps", 30.0),
    )

    # Primary Beacon
    b_cfg = cfg.get("beacon", {})
    init_pos = b_cfg.get("initial_pos_deg", [0.0, 0.0])
    traj_cfg = b_cfg.get("trajectory", {"type": "sinusoidal"})
    traj = create_trajectory(traj_cfg)

    blink_cfg = b_cfg.get("blinking", {})
    primary_beacon = Beacon(
        beacon_id=b_cfg.get("id", 1),
        name=b_cfg.get("name", "primary_beacon"),
        is_primary=True,
        initial_pos_rad=(deg2rad(init_pos[0]), deg2rad(init_pos[1])),
        intensity=b_cfg.get("intensity", 255.0),
        radius_pixels=b_cfg.get("radius_pixels", 7.0),
        color_bgr=tuple(b_cfg.get("color_bgr", [255, 255, 230])),
        trajectory=traj,
        blinking_enabled=blink_cfg.get("enabled", False),
        blinking_frequency_hz=blink_cfg.get("frequency_hz", 2.0),
        blinking_duty_cycle=blink_cfg.get("duty_cycle", 0.5),
    )
    env.add_beacon(primary_beacon)

    # Distractors
    for d_cfg in b_cfg.get("distractors", []):
        d_pos = d_cfg.get("initial_pos_deg", [5.0, -3.0])
        d_traj_cfg = d_cfg.get("trajectory", {"type": "linear"})
        d_traj = create_trajectory(d_traj_cfg)
        distractor = Beacon(
            beacon_id=d_cfg.get("id", 2),
            name=d_cfg.get("name", "distractor"),
            is_primary=False,
            initial_pos_rad=(deg2rad(d_pos[0]), deg2rad(d_pos[1])),
            intensity=d_cfg.get("intensity", 180.0),
            radius_pixels=d_cfg.get("radius_pixels", 4.0),
            color_bgr=tuple(d_cfg.get("color_bgr", [100, 100, 255])),
            trajectory=d_traj,
        )
        env.add_beacon(distractor)

    return env, cam
