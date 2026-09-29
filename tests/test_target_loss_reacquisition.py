"""
Integration test for temporary target loss, dead-reckoning coasting, and reacquisition.
"""

import math
import pytest
import numpy as np

from simulation.coordinate_system import deg2rad, rad2deg
from simulation.trajectory import LinearTrajectory
from simulation.beacon import Beacon
from simulation.camera import VirtualCamera
from simulation.environment import VirtualEnvironment
from vision.classical_detector import ClassicalBeaconDetector
from control.camera_controller import CameraGimbalController
from tracking.tracker import PATTracker
from tracking.state_machine import PATState


def test_target_loss_and_reacquisition():
    dt = 1.0 / 30.0
    camera = VirtualCamera(resolution=(1280, 720), fov_horizontal_deg=20.0, fov_vertical_deg=12.0)
    camera.set_orientation(0.0, 0.0)

    # Beacon moving at 1.0 deg/s
    traj = LinearTrajectory(x0=0.0, y0=0.0, vx=1.0, vy=0.0)
    beacon = Beacon(
        beacon_id=1,
        initial_pos_rad=(0.0, 0.0),
        intensity=255.0,
        trajectory=traj,
    )

    env = VirtualEnvironment(width_deg=40.0, height_deg=25.0)
    env.add_beacon(beacon)

    detector = ClassicalBeaconDetector(binary_threshold=150)
    tracker = PATTracker(camera=camera, prediction_horizon_steps=5)
    controller = CameraGimbalController(camera=camera, kp_pan=2.0, kp_tilt=2.0)

    states_recorded = []

    for step in range(80):
        t = step * dt
        env.update(t, dt)

        # Artificial occlusion between step 25 and step 33 (8 frames blind)
        is_occluded = (25 <= step <= 33)
        beacon.occluded = is_occluded

        bg = env.generate_camera_background(camera)
        frame = camera.render_frame(env.beacons, t, base_canvas=bg)

        # Vision detect
        det_result = detector.detect(frame, timestamp=t)
        primary_det = det_result.primary if not is_occluded else None

        # Track & State Machine
        tracker_out = tracker.process_frame(primary_det, sim_time=t, dt=dt)
        states_recorded.append(tracker_out.state)

        # Closed-loop control
        if tracker_out.search_rate_rad_s is not None:
            camera.set_command_rate(
                tracker_out.search_rate_rad_s[0], tracker_out.search_rate_rad_s[1]
            )
        else:
            controller.track_angular_error(
                tracker_out.cmd_delta_az_rad, tracker_out.cmd_delta_el_rad, dt
            )

        camera.step(dt)

    # Check state history:
    # 1. Started SEARCHING / ACQUIRING
    assert PATState.ACQUIRING in states_recorded or PATState.TRACKING in states_recorded
    # 2. Entered UNCERTAIN during occlusion
    assert PATState.UNCERTAIN in states_recorded[25:35]
    # 3. Recovered to TRACKING after occlusion ended
    assert states_recorded[-1] == PATState.TRACKING
