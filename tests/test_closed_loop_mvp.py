"""
End-to-End closed-loop tracking MVP verification test.

Verifies:
Moving optical beacon in virtual environment
-> Virtual camera captures viewport
-> Classical CV detector identifies beacon
-> Pixel-to-angular error calculated
-> PID controller steers camera gimbal
-> Beacon becomes centered within boresight dead-zone
-> Camera continuously follows moving target
"""

import math
import pytest
import numpy as np

from simulation.coordinate_system import deg2rad, rad2deg
from simulation.trajectory import SinusoidalTrajectory
from simulation.beacon import Beacon
from simulation.camera import VirtualCamera
from simulation.environment import VirtualEnvironment
from vision.classical_detector import ClassicalBeaconDetector
from control.camera_controller import CameraGimbalController


def test_closed_loop_mvp_convergence():
    dt = 1.0 / 30.0  # 30 FPS simulation
    sim_steps = 120

    # 1. Setup Camera
    camera = VirtualCamera(
        resolution=(1280, 720),
        fov_horizontal_deg=25.0,
        fov_vertical_deg=14.0,
        max_pan_speed_deg_s=30.0,
        max_tilt_speed_deg_s=30.0,
    )
    camera.set_orientation(0.0, 0.0)

    # 2. Setup Beacon initially offset at (3.5 deg, 2.0 deg) moving horizontally and waving
    traj = SinusoidalTrajectory(
        x0=3.5,
        y0=2.0,
        vx=0.5,              # 0.5 deg/s drift
        amplitude_y=0.4,
        frequency_hz=0.1,
    )
    beacon = Beacon(
        beacon_id=1,
        initial_pos_rad=(deg2rad(3.5), deg2rad(2.0)),
        intensity=255.0,
        radius_pixels=7.0,
        trajectory=traj,
    )

    env = VirtualEnvironment(width_deg=60.0, height_deg=35.0, num_stars=50)
    env.add_beacon(beacon)

    # 3. Setup Detector and Controller
    detector = ClassicalBeaconDetector(binary_threshold=150)
    controller = CameraGimbalController(
        camera=camera,
        kp_pan=2.5,
        ki_pan=0.8,
        kd_pan=0.15,
        kp_tilt=2.5,
        ki_tilt=0.8,
        kd_tilt=0.15,
        dead_zone_deg=0.04,
    )

    initial_error_deg = None
    final_errors = []

    # Run closed-loop simulation
    for step in range(sim_steps):
        t = step * dt

        # Update environment physics
        env.update(t, dt)

        # Render camera viewport frame
        bg = env.generate_camera_background(camera)
        frame = camera.render_frame(env.beacons, t, base_canvas=bg)

        # Vision detection
        det_result = detector.detect(frame, timestamp=t)
        assert det_result.success is True, f"Detection failed at step {step}"
        primary_det = det_result.primary

        # Calculate tracking error and command gimbal
        controller.track_pixel_error(primary_det.center_u, primary_det.center_v, dt)

        # Step camera physical kinematics
        camera.step(dt)

        # Ground truth error
        gt_delta_az, gt_delta_el, _, _, in_fov = camera.project_target(
            beacon.pos[0], beacon.pos[1]
        )
        total_error_deg = rad2deg(math.hypot(gt_delta_az, gt_delta_el))

        if initial_error_deg is None:
            initial_error_deg = total_error_deg

        if step > 85:
            final_errors.append(total_error_deg)

    # Initial error should be around hypot(3.5, 2.0) ~= 4.03 deg
    assert initial_error_deg > 3.0

    # Camera should have converged onto target: mean error in final 35 steps should be < 0.30 deg (coarse lock threshold)
    mean_final_error = float(np.mean(final_errors))
    assert mean_final_error < 0.30, f"Expected final error < 0.30 deg, got {mean_final_error:.3f} deg"
    assert bool(in_fov) is True
