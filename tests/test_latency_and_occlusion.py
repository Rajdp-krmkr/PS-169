"""
Unit tests for ControlDelayQueue and DynamicOccluder line-of-sight blockages.
"""

import math
import numpy as np
import pytest

from control.latency import ControlDelayQueue
from disturbances.occluder import DynamicOccluder, OccluderManager
from simulation.camera import VirtualCamera


def test_control_delay_queue():
    # 1. Zero delay -> immediate passthrough
    queue0 = ControlDelayQueue(delay_frames=0)
    out_p, out_t = queue0.step(1.5, -0.8)
    assert out_p == 1.5
    assert out_t == -0.8

    # 2. 2-frame delay -> FIFO buffering
    queue2 = ControlDelayQueue(delay_frames=2)
    # Frame 0: input (1.0, 2.0) -> outputs (0.0, 0.0)
    p0, t0 = queue2.step(1.0, 2.0)
    assert (p0, t0) == (0.0, 0.0)

    # Frame 1: input (3.0, 4.0) -> outputs (0.0, 0.0)
    p1, t1 = queue2.step(3.0, 4.0)
    assert (p1, t1) == (0.0, 0.0)

    # Frame 2: input (5.0, 6.0) -> outputs (1.0, 2.0) (buffered from frame 0)
    p2, t2 = queue2.step(5.0, 6.0)
    assert (p2, t2) == (1.0, 2.0)

    # Frame 3: input (7.0, 8.0) -> outputs (3.0, 4.0) (buffered from frame 1)
    p3, t3 = queue2.step(7.0, 8.0)
    assert (p3, t3) == (3.0, 4.0)


def test_dynamic_occluder_blocking_and_attenuation():
    # Cloud centered at (0.010, 0.0) with radius 0.005 rad, moving at +0.010 rad/s
    occ = DynamicOccluder(
        initial_pos_rad=(0.010, 0.0),
        velocity_rad_s=(0.010, 0.0),
        radius_rad=0.005,
        opacity=1.0,
    )

    # At t=0, target at (0.010, 0.0) is directly inside the cloud
    assert occ.is_occluding(0.010, 0.0) is True
    assert occ.get_transmission_factor(0.010, 0.0) == 0.0  # Fully blocked

    # Target outside cloud at (0.020, 0.0) (distance 0.010 > radius 0.005)
    assert occ.is_occluding(0.020, 0.0) is False
    assert occ.get_transmission_factor(0.020, 0.0) == 1.0

    # Step occluder by 1.0 second: pos moves to 0.010 + 0.010 = 0.020 rad
    occ.step(1.0)
    assert math.isclose(occ.az_rad, 0.020, abs_tol=1e-5)
    # Now target at (0.020, 0.0) is occluded!
    assert occ.is_occluding(0.020, 0.0) is True


def test_occluder_manager():
    mgr = OccluderManager()
    occ1 = DynamicOccluder(initial_pos_rad=(0.0, 0.0), radius_rad=0.004, opacity=0.8)
    mgr.add_occluder(occ1)

    blocked, trans = mgr.check_occlusion(0.0, 0.0)
    assert blocked is True
    assert trans < 0.3
