"""
Integration test for full closed-loop tracking using the Hybrid Detection Pipeline.
"""

import math
import numpy as np
import pytest

from simulation.coordinate_system import deg2rad, rad2deg
from simulation.trajectory import SinusoidalTrajectory
from simulation.beacon import Beacon
from simulation.camera import VirtualCamera
from simulation.environment import VirtualEnvironment
from vision.classical_detector import ClassicalBeaconDetector
from vision.yolo_detector import YOLODetector
from vision.fusion import DetectionFusion
from control.camera_controller import CameraGimbalController
from tracking.tracker import PATTracker
from tracking.state_machine import PATState


def test_hybrid_tracking_closed_loop():
    dt = 1.0 / 30.0
    camera = VirtualCamera(resolution=(1280, 720), fov_horizontal_deg=20.0, fov_vertical_deg=11.25)
    camera.set_orientation(0.0, 0.0)

    # Beacon moving sinusoidally
    traj = SinusoidalTrajectory(x0=1.5, y0=1.0, vx=0.8, amplitude_y=0.5, frequency_hz=0.2)
    beacon = Beacon(beacon_id=1, initial_pos_rad=(deg2rad(1.5), deg2rad(1.0)), intensity=255.0, trajectory=traj)

    env = VirtualEnvironment(width_deg=40.0, height_deg=25.0)
    env.add_beacon(beacon)

    detector_cv = ClassicalBeaconDetector(binary_threshold=140)
    detector_ai = YOLODetector(model_path="models/non_existent.onnx")  # Operates in graceful fallback mode
    fusion = DetectionFusion()
    tracker = PATTracker(camera=camera, prediction_horizon_steps=5)
    controller = CameraGimbalController(camera=camera, kp_pan=2.0, kp_tilt=2.0)

    final_errors = []

    for step in range(70):
        t = step * dt
        env.update(t, dt)

        bg = env.generate_camera_background(camera)
        frame = camera.render_frame(env.beacons, t, base_canvas=bg)

        # Multi-modal detection
        res_cv = detector_cv.detect(frame, timestamp=t)
        res_ai = detector_ai.detect(frame, timestamp=t)
        res_fused = fusion.fuse(res_cv, res_ai, timestamp=t)

        tracker_out = tracker.process_frame(res_fused.primary, sim_time=t, dt=dt)

        if tracker_out.search_rate_rad_s is not None:
            camera.set_command_rate(
                tracker_out.search_rate_rad_s[0], tracker_out.search_rate_rad_s[1]
            )
        else:
            controller.track_angular_error(
                tracker_out.cmd_delta_az_rad, tracker_out.cmd_delta_el_rad, dt
            )

        camera.step(dt)

        # Ground truth error
        gt_az, gt_el, _, _, in_fov = camera.project_target(beacon.pos[0], beacon.pos[1])
        err = rad2deg(math.hypot(gt_az, gt_el))
        if step > 45:
            final_errors.append(err)

    # State should be TRACKING and mean error < 0.35 deg
    assert tracker_out.state == PATState.TRACKING
    assert np.mean(final_errors) < 0.35
